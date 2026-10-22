/*
 * ksw_align2() for many alignments at once, with NEON (aarch64).
 *
 * ksw_u8() aligns one query (striped, Farrar) at a time; its lazy-F loop and
 * short dependency chains leave most of the vector units idle. Here 16
 * independent alignments are computed at once, one per byte lane, cell by
 * cell along the query. The results are identical to those of ksw_align2(),
 * including the details which follow from the striped layout:
 *
 *  - The query is padded to a multiple of 16 positions, which score 0 (as in
 *    the striped query profile); they count for the maximum of a row.
 *  - E (deletions) is computed from the H which includes the F (insertions)
 *    carried within a stripe of the query (slen consecutive positions) but
 *    not those carried across stripes by the lazy-F loop. So two F values
 *    are kept: one reset at the start of each stripe, and the full one.
 *  - The row maxima, the second best score and its position are recorded as
 *    with the b array of ksw_u8().
 *
 * Alignments outside this scope (16-bit scores, a saturated score of 255,
 * very long sequences) are passed to ksw_align2().
 *
 * Copyright and license as for ksw.cpp.
 */

#if defined(__aarch64__)

#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include "ksw.h"

namespace {

const int W = 16;   // alignments per vector

struct Problem {
	const uint8_t *q; int qlen;
	const uint8_t *t; int tlen;
	int minsc, endsc;   // as in ksw_u8()
	kswr_t r;           // score, te, qe, score2, te2
};

// Score table: tbl[t + 5*q] = mat[t*5 + q] + shift; tbl[t + 25] = shift for
// the padding of the query
struct Scoring {
	uint8x16x2_t tbl;
	uint8_t shift; int qmax;
	uint8x16_t e_del, oe_del, e_ins, oe_ins;
	uint8_t e_ins_;
};

// Transposes a 16x16 matrix of bytes
inline void transpose16(uint8x16_t *r)
{
	for (int k = 0; k < 16; k += 2) {
		uint8x16_t a = vtrn1q_u8(r[k], r[k + 1]), b = vtrn2q_u8(r[k], r[k + 1]);
		r[k] = a; r[k + 1] = b;
	}
	for (int k = 0; k < 16; k += 4)
		for (int c = 0; c < 2; ++c) {
			uint16x8_t x = vreinterpretq_u16_u8(r[k + c]), y = vreinterpretq_u16_u8(r[k + c + 2]);
			r[k + c] = vreinterpretq_u8_u16(vtrn1q_u16(x, y)); r[k + c + 2] = vreinterpretq_u8_u16(vtrn2q_u16(x, y));
		}
	for (int k = 0; k < 16; k += 8)
		for (int c = 0; c < 4; ++c) {
			uint32x4_t x = vreinterpretq_u32_u8(r[k + c]), y = vreinterpretq_u32_u8(r[k + c + 4]);
			r[k + c] = vreinterpretq_u8_u32(vtrn1q_u32(x, y)); r[k + c + 4] = vreinterpretq_u8_u32(vtrn2q_u32(x, y));
		}
	for (int c = 0; c < 8; ++c) {
		uint64x2_t x = vreinterpretq_u64_u8(r[c]), y = vreinterpretq_u64_u8(r[c + 8]);
		r[c] = vreinterpretq_u8_u64(vtrn1q_u64(x, y)); r[c + 8] = vreinterpretq_u8_u64(vtrn2q_u64(x, y));
	}
}

// The alignment of up to W problems, as ksw_u8() would each of them, row
// (target position) by row
struct Group {
	Problem *p; int n;
	int Q, T;
	bool masked;   // whether the problems have different padded query lengths
	std::vector<uint8_t> q5, valid, keep, fdec, Hbuf0, Hbuf1, E, trow, rowmax;
	uint8_t *H0, *H1;
	uint8_t slen[W], minsc[W], endsc[W], subo[W], stop[W];
	uint16_t tlen[W];
	uint8x16_t live, gmax, qe, prev_rec, prev_score;
	uint16x8_t te_lo, te_hi;
	uint16x8_t rows_lo, rows_hi;   // rows processed, per alignment

	void init(Problem *p_, int n_, const Scoring &sc) {
		p = p_; n = n_; Q = 0; T = 0;
		for (int k = 0; k < W; ++k) {
			int ql = k < n ? p[k].qlen : 0;
			slen[k] = (ql + 15) / 16;
			if (16 * slen[k] > Q) Q = 16 * slen[k];
			tlen[k] = k < n ? p[k].tlen : 0;
			if (tlen[k] > T) T = tlen[k];
			subo[k] = k < n && p[k].minsc <= 255 ? 0xff : 0;
			minsc[k] = subo[k] ? p[k].minsc : 0;
			stop[k] = k < n && p[k].endsc <= 255 ? 0xff : 0;
			endsc[k] = stop[k] ? p[k].endsc : 0;
		}
		masked = false;
		for (int k = 1; k < n; ++k) if (slen[k] != slen[0]) masked = true;
		size_t qs = (size_t) (Q + 1) * W, ts = (size_t) (T + 15) / 16 * 16 * W;
		if (q5.size() < qs) { q5.resize(qs); valid.resize(qs); keep.resize(qs); fdec.resize(qs); Hbuf0.resize(qs); Hbuf1.resize(qs); E.resize(qs); }
		if (trow.size() < ts) { trow.resize(ts); rowmax.resize(ts); }
		// query columns: 5*q (25 for padding), whether the position exists in
		// the striped layout, and whether the F within a stripe carries on to
		// the next position (if not, it is decreased by 255, i.e. reset)
		for (int j = 0; j < Q; ++j)
			for (int k = 0; k < W; ++k) {
				int ql = k < n ? p[k].qlen : 0, in = j < 16 * slen[k];
				int carry = in && (j + 1) % slen[k] != 0;
				q5[j * W + k] = j < ql ? 5 * p[k].q[j] : 25;
				valid[j * W + k] = in ? 0xff : 0;
				keep[j * W + k] = carry ? 0xff : 0;
				fdec[j * W + k] = carry ? sc.e_ins_ : 0xff;
			}
		// the targets, by row (transposed by blocks of 16 positions)
		for (int i0 = 0; i0 < T; i0 += 16) {
			uint8x16_t r[16];
			for (int k = 0; k < W; ++k) {
				if (i0 + 16 <= tlen[k]) r[k] = vld1q_u8(p[k].t + i0);
				else {
					uint8_t tmp[16] = {0};
					for (int c = 0; i0 + c < tlen[k] && c < 16; ++c) tmp[c] = p[k].t[i0 + c];
					r[k] = vld1q_u8(tmp);
				}
			}
			transpose16(r);
			for (int c = 0; c < 16; ++c) vst1q_u8(&trow[(size_t) (i0 + c) * W], r[c]);
		}
		rows_lo = rows_hi = vdupq_n_u16(0);
		H0 = Hbuf0.data(); H1 = Hbuf1.data();
		memset(H0, 0, (size_t) Q * W);
		memset(E.data(), 0, (size_t) Q * W);
		live = vdupq_n_u8(0xff);   // not stopped
		gmax = qe = prev_rec = prev_score = vdupq_n_u8(0);
		te_lo = te_hi = vdupq_n_u16(0xffff);
	}

	// The lanes taking part in row i (not stopped, and i < tlen), or 0
	uint8x16_t active(int i) const {
		uint16x8_t vi = vdupq_n_u16(i);
		uint8x16_t inrow = vcombine_u8(vmovn_u16(vcltq_u16(vi, vld1q_u16(tlen))), vmovn_u16(vcltq_u16(vi, vld1q_u16(tlen + 8))));
		return vandq_u8(live, inrow);
	}

	// After row i, with rmax the maxima of the row
	void end_row(int i, uint8x16_t act, uint8x16_t rmax) {
		// b array of ksw_u8(): a new entry, or the last one replaced when it
		// is in the previous row and has a lower score
		uint8x16_t ge = vandq_u8(vandq_u8(vcgeq_u8(rmax, vld1q_u8(minsc)), vld1q_u8(subo)), act);
		uint8x16_t repl = vandq_u8(vandq_u8(ge, prev_rec), vcgtq_u8(rmax, prev_score));
		uint8x16_t add = vbicq_u8(ge, prev_rec);
		uint8x16_t rec = vorrq_u8(repl, add);
		vst1q_u8(&rowmax[i * W], vandq_u8(rec, rmax));
		if (i > 0) vst1q_u8(&rowmax[(i - 1) * W], vbicq_u8(vld1q_u8(&rowmax[(i - 1) * W]), repl));
		prev_score = vbslq_u8(rec, rmax, prev_score);
		prev_rec = rec;

		// best score so far, and stopping
		uint8x16_t up = vandq_u8(vcgtq_u8(rmax, gmax), act);
		gmax = vbslq_u8(up, rmax, gmax);
		if (vmaxvq_u8(up)) {
			// the query end: the first position with the maximum (the F
			// carried from other stripes is lower than the maximum, so the
			// final H can be used)
			uint8x16_t todo = up;
			for (int j = 0; j < Q; ++j) {
				uint8x16_t eq = vandq_u8(vceqq_u8(vld1q_u8(H1 + j * W), rmax), todo);
				qe = vbslq_u8(eq, vdupq_n_u8(j), qe);
				todo = vbicq_u8(todo, eq);
				if ((j & 7) == 7 && vmaxvq_u8(todo) == 0) break;
			}
			uint16x8_t vi = vdupq_n_u16(i);
			uint16x8_t up_lo = vmovl_u8(vget_low_u8(up)), up_hi = vmovl_u8(vget_high_u8(up));
			up_lo = vorrq_u16(up_lo, vshlq_n_u16(up_lo, 8)); up_hi = vorrq_u16(up_hi, vshlq_n_u16(up_hi, 8));
			te_lo = vbslq_u16(up_lo, vi, te_lo); te_hi = vbslq_u16(up_hi, vi, te_hi);
			uint8x16_t halt = vorrq_u8(vcgeq_u8(gmax, vdupq_n_u8(255 - satshift)),
			                           vandq_u8(vcgeq_u8(gmax, vld1q_u8(endsc)), vld1q_u8(stop)));
			live = vbicq_u8(live, vandq_u8(up, halt));
		}

		uint8x16_t one = vandq_u8(act, vdupq_n_u8(1));
		rows_lo = vaddw_u8(rows_lo, vget_low_u8(one)); rows_hi = vaddw_u8(rows_hi, vget_high_u8(one));
		uint8_t *tmp = H0; H0 = H1; H1 = tmp;
	}

	void finish(const Scoring &sc) {
		uint8_t g[W], qq[W];
		uint16_t te[W];
		vst1q_u8(g, gmax); vst1q_u8(qq, qe);
		vst1q_u16(te, te_lo); vst1q_u16(te + 8, te_hi);
		// the second best: the highest b entry outside [te - val, te + val]
		// (the first one if several), as in ksw_u8()
		int16_t low[W], high[W];
		int maxrows = 0;
		uint16_t rows[W];
		vst1q_u16(rows, rows_lo); vst1q_u16(rows + 8, rows_hi);
		for (int k = 0; k < W; ++k) {
			int score = g[k], tk = te[k] == 0xffff ? -1 : te[k];
			int val = (score + sc.qmax - 1) / sc.qmax;
			low[k] = tk - val; high[k] = tk + val;
			if (k < n && rows[k] > maxrows) maxrows = rows[k];
		}
		int16x8_t vlow[2] = { vld1q_s16(low), vld1q_s16(low + 8) }, vhigh[2] = { vld1q_s16(high), vld1q_s16(high + 8) };
		uint16x8_t vrows[2] = { rows_lo, rows_hi }, s2[2], te2[2];
		for (int h = 0; h < 2; ++h) s2[h] = vdupq_n_u16(0), te2[h] = vdupq_n_u16(0xffff);
		for (int row = 0; row < maxrows; ++row) {
			uint8x16_t v8 = vld1q_u8(&rowmax[(size_t) row * W]);
			uint16x8_t v[2] = { vmovl_u8(vget_low_u8(v8)), vmovl_u8(vget_high_u8(v8)) };
			int16x8_t vr = vdupq_n_s16(row);
			for (int h = 0; h < 2; ++h) {
				uint16x8_t out = vorrq_u16(vcltq_s16(vr, vlow[h]), vcgtq_s16(vr, vhigh[h]));
				uint16x8_t upd = vandq_u16(vandq_u16(out, vcltq_u16(vreinterpretq_u16_s16(vr), vrows[h])), vcgtq_u16(v[h], s2[h]));
				s2[h] = vbslq_u16(upd, v[h], s2[h]);
				te2[h] = vbslq_u16(upd, vreinterpretq_u16_s16(vr), te2[h]);
			}
		}
		uint16_t sc2[W], t2[W];
		vst1q_u16(sc2, s2[0]); vst1q_u16(sc2 + 8, s2[1]);
		vst1q_u16(t2, te2[0]); vst1q_u16(t2 + 8, te2[1]);
		for (int k = 0; k < n; ++k) {
			kswr_t r = g_defr;
			r.score = g[k] + sc.shift < 255 ? g[k] : 255;
			r.te = te[k] == 0xffff ? -1 : te[k];
			if (r.score != 255) {
				r.qe = qq[k];   // 0 if no cell was positive, as in ksw_u8()
				if (sc2[k]) r.score2 = sc2[k], r.te2 = t2[k];
			}
			p[k].r = r;
		}
	}

	uint8_t satshift;   // gmax + shift >= 255 stops
};

// One row for NG groups with the same padded query length Q: returns the
// row maxima of H. With masked, the query positions beyond the padded length
// of an alignment are kept at 0.
template<int NG, bool masked>
inline void align_rows(Group **g, const uint8x16_t *t, const Scoring &sc, uint8x16_t *rmax)
{
	const int Q = g[0]->Q;
	const uint8x16_t zero = vdupq_n_u8(0), shift = vdupq_n_u8(sc.shift);
	uint8x16_t d[NG], fs[NG], ff[NG];
	const uint8_t *__restrict q5[NG], *__restrict valid[NG], *__restrict keep[NG], *__restrict fdec[NG], *__restrict H0[NG];
	uint8_t *__restrict E[NG], *__restrict H1[NG];
	for (int x = 0; x < NG; ++x) {
		d[x] = fs[x] = ff[x] = rmax[x] = zero;
		q5[x] = g[x]->q5.data(); valid[x] = g[x]->valid.data(); keep[x] = g[x]->keep.data();
		fdec[x] = g[x]->fdec.data(); H0[x] = g[x]->H0; H1[x] = g[x]->H1; E[x] = g[x]->E.data();
	}
	for (int j = 0; j < Q; ++j) {
		const size_t o = (size_t) j * W;
		for (int x = 0; x < NG; ++x) {
			// hd = max{H(i-1,j-1) + S(i,j), E(i,j)}; H(i,j) = max{hd, F}.
			// The next F is max{F - e_ins, H(i,j) - oe_ins}, which equals
			// max{F - e_ins, hd - oe_ins} as oe_ins >= e_ins (saturating
			// arithmetic is monotonic): this keeps H out of the F chains. The
			// full F is never below the F within the stripe, so the same
			// holds for it.
			uint8x16_t s = vqtbl2q_u8(sc.tbl, vaddq_u8(t[x], vld1q_u8(q5[x] + o)));
			uint8x16_t hd = vqsubq_u8(vqaddq_u8(d[x], s), shift);
			uint8x16_t e = vld1q_u8(E[x] + o);
			hd = vmaxq_u8(hd, e);
			if (masked) hd = vandq_u8(hd, vld1q_u8(valid[x] + o));
			uint8x16_t gi = vqsubq_u8(hd, sc.oe_ins);
			uint8x16_t h = vmaxq_u8(hd, fs[x]);   // H(i,j) without the F from other stripes
			rmax[x] = vmaxq_u8(rmax[x], h);
			vst1q_u8(E[x] + o, vmaxq_u8(vqsubq_u8(e, sc.e_del), vqsubq_u8(h, sc.oe_del)));
			vst1q_u8(H1[x] + o, vmaxq_u8(h, ff[x]));
			ff[x] = vmaxq_u8(vqsubq_u8(ff[x], sc.e_ins), gi);
			fs[x] = vmaxq_u8(vqsubq_u8(fs[x], vld1q_u8(fdec[x] + o)), vandq_u8(gi, vld1q_u8(keep[x] + o)));
			d[x] = vld1q_u8(H0[x] + o);
		}
	}
}

// Aligns the groups (one, or two with the same padded query length) to the end
template<int NG>
void align_groups(Group **g, const Scoring &sc)
{
	bool masked = false;
	int T = 0;
	for (int x = 0; x < NG; ++x) { masked |= g[x]->masked; T = std::max(T, g[x]->T); }
	for (int i = 0; i < T; ++i) {
		uint8x16_t act[NG], t[NG], rmax[NG];
		bool any = false;
		for (int x = 0; x < NG; ++x) {
			act[x] = i < g[x]->T ? g[x]->active(i) : vdupq_n_u8(0);
			any |= vmaxvq_u8(act[x]) != 0;
			t[x] = vld1q_u8(&g[x]->trow[(size_t) std::min(i, g[x]->T - 1) * W]);
		}
		if (!any) break;
		if (masked) align_rows<NG, true>(g, t, sc, rmax);
		else align_rows<NG, false>(g, t, sc, rmax);
		for (int x = 0; x < NG; ++x) if (i < g[x]->T) g[x]->end_row(i, act[x], rmax[x]);
	}
}

thread_local Group groups[2];

// Aligns the problems by groups of W of similar lengths, two groups at once
void align_u8(std::vector<Problem> &probs, const Scoring &sc)
{
	std::vector<int> order(probs.size());
	for (size_t k = 0; k < order.size(); ++k) order[k] = k;
	// by padded query length, then by target length
	std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
		int qa = (probs[a].qlen + 15) / 16, qb = (probs[b].qlen + 15) / 16;
		return qa != qb ? qa > qb : probs[a].tlen > probs[b].tlen;
	});
	std::vector<Problem> sorted(probs.size());
	for (size_t k = 0; k < order.size(); ++k) sorted[k] = probs[order[k]];
	size_t s = 0;
	while (s < sorted.size()) {
		Group *g[2] = { &groups[0], &groups[1] };
		int n0 = std::min((size_t) W, sorted.size() - s);
		g[0]->satshift = g[1]->satshift = sc.shift;
		g[0]->init(&sorted[s], n0, sc);
		size_t s1 = s + n0;
		int n1 = std::min((size_t) W, sorted.size() - s1);
		if (n1 > 0) {
			g[1]->init(&sorted[s1], n1, sc);
			if (g[1]->Q == g[0]->Q) {
				align_groups<2>(g, sc);
				g[0]->finish(sc); g[1]->finish(sc);
				s = s1 + n1;
				continue;
			}
		}
		align_groups<1>(g, sc);
		g[0]->finish(sc);
		s = s1;
	}
	for (size_t k = 0; k < order.size(); ++k) probs[order[k]].r = sorted[k].r;
}

} // namespace

void ksw_align2_batch(int n, int *qlen, uint8_t **query, int *tlen, uint8_t **target,
                      int m, const int8_t *mat, int o_del, int e_del, int o_ins, int e_ins,
                      int *xtra, kswr_t *out)
{
	Scoring sc;
	int8_t mn = 127, mx = 0;
	for (int a = 0; a < m * m; ++a) { if (mat[a] < mn) mn = mat[a]; if (mat[a] > mx) mx = mat[a]; }
	sc.shift = (uint8_t) (256 - mn);
	sc.qmax = mx;
	uint8_t tbl[32] = {0};
	for (int t = 0; t < 5; ++t) {
		for (int q = 0; q < 5; ++q) tbl[t + 5 * q] = (uint8_t) (mat[t * m + q] + sc.shift);
		tbl[t + 25] = sc.shift;
	}
	sc.tbl.val[0] = vld1q_u8(tbl); sc.tbl.val[1] = vld1q_u8(tbl + 16);
	sc.e_del = vdupq_n_u8(e_del); sc.oe_del = vdupq_n_u8(o_del + e_del);
	sc.e_ins = vdupq_n_u8(e_ins); sc.oe_ins = vdupq_n_u8(o_ins + e_ins);
	sc.e_ins_ = e_ins;

	// forward alignments
	std::vector<Problem> fwd;
	std::vector<int> fwd_id;
	for (int k = 0; k < n; ++k) {
		bool ok = m == 5 && (xtra[k] & KSW_XBYTE) && qlen[k] > 0 && qlen[k] <= 240 && tlen[k] > 0 && tlen[k] <= 32000
			&& !((xtra[k] & KSW_XSUBO) && (xtra[k] & 0xffff) == 0);
		if (!ok) {
			out[k] = ksw_align2(qlen[k], query[k], tlen[k], target[k], m, mat, o_del, e_del, o_ins, e_ins, xtra[k], 0);
			continue;
		}
		Problem pr;
		pr.q = query[k]; pr.qlen = qlen[k]; pr.t = target[k]; pr.tlen = tlen[k];
		pr.minsc = (xtra[k] & KSW_XSUBO) ? xtra[k] & 0xffff : 0x10000;
		pr.endsc = (xtra[k] & KSW_XSTOP) ? xtra[k] & 0xffff : 0x10000;
		fwd.push_back(pr); fwd_id.push_back(k);
	}
	align_u8(fwd, sc);

	// reverse alignments, for the start positions
	std::vector<Problem> rev;
	std::vector<int> rev_id;
	std::vector<std::vector<uint8_t> > seqs;
	for (size_t f = 0; f < fwd.size(); ++f) {
		int k = fwd_id[f];
		kswr_t r = fwd[f].r;
		out[k] = r;
		if ((xtra[k] & KSW_XSTART) == 0 || ((xtra[k] & KSW_XSUBO) && r.score < (xtra[k] & 0xffff))) continue;
		if (r.score == 255 || r.qe < 0 || r.te < 0) {   // not expected with KSW_XBYTE; as ksw_align2()
			out[k] = ksw_align2(qlen[k], query[k], tlen[k], target[k], m, mat, o_del, e_del, o_ins, e_ins, xtra[k], 0);
			continue;
		}
		// as ksw_align2(): the query and target ends reversed, the rest of the
		// target as it is
		seqs.push_back(std::vector<uint8_t>(query[k], query[k] + r.qe + 1));
		std::reverse(seqs.back().begin(), seqs.back().end());
		seqs.push_back(std::vector<uint8_t>(target[k], target[k] + tlen[k]));
		std::reverse(seqs.back().begin(), seqs.back().begin() + r.te + 1);
		Problem pr;
		pr.qlen = r.qe + 1; pr.tlen = tlen[k];
		pr.minsc = 0x10000; pr.endsc = r.score;
		rev.push_back(pr); rev_id.push_back(k);
	}
	for (size_t v = 0; v < rev.size(); ++v) {
		rev[v].q = seqs[2 * v].data();
		rev[v].t = seqs[2 * v + 1].data();
	}
	align_u8(rev, sc);
	for (size_t v = 0; v < rev.size(); ++v) {
		kswr_t &r = out[rev_id[v]];
		if (r.score == rev[v].r.score) {
			r.tb = r.te - rev[v].r.te;
			r.qb = r.qe - rev[v].r.qe;
		}
	}
}

#endif
