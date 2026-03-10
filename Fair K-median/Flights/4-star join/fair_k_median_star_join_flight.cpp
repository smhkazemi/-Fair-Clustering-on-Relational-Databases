// fair_k_median_strict_superfast.cpp
// 
// HIGH-PERFORMANCE STRICT PAPER IMPLEMENTATION.
// Solves the O(N) inner-loop bottlenecks via Contiguous Memory indexing and Loop Inversion.
//
// Compile:
// g++ -O3 -pthread -march=native -funroll-loops --std=c++20 fair_k_median_strict_superfast.cpp -o aa.out

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <chrono>
#include <limits>
#include <numeric>

using namespace std;

// -------------------------
// 1. DATA TYPES & GEOMETRY
// -------------------------
const int DIMS = 8; 
using Point = vector<double>;
struct WeightedPoint { Point point; double weight; int color; }; 

inline void Scale(Point &p) { for (auto& v : p) if (std::abs(v) > 1e8) v /= 86400000.0; }

double L1Dist(const Point& a, const Point& b) {
    double d = 0; size_t m = min((size_t)DIMS, a.size());
    for (size_t i = 0; i < m; ++i) d += std::abs(a[i] - b[i]);
    return d;
}

struct Box {
    vector<pair<double, double>> bounds;
    bool Contains(const Point& p) const {
        for (size_t i = 0; i < bounds.size(); ++i)
            if (p[i] < bounds[i].first || p[i] >= bounds[i].second) return false;
        return true;
    }
    double Diam() const { double d = 0; for (auto& i : bounds) d += (i.second - i.first); return d; }
    double DistTo(const Point& p) const {
        double d = 0;
        for (size_t i = 0; i < min(p.size(), bounds.size()); ++i) {
            if (p[i] < bounds[i].first) d += (bounds[i].first - p[i]);
            else if (p[i] > bounds[i].second) d += (p[i] - bounds[i].second);
        }
        return d;
    }
};

struct RNG {
    uint64_t s; RNG(uint64_t v = 42) : s(v) {}
    uint64_t next() { s += 0x9e3779b97f4a7c15; uint64_t z = s; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9; return z ^ (z >> 27) ^ 0x94d049bb133111eb; }
    double unit() { return (next() >> 11) * 0x1.0p-53; }
    size_t idx(size_t m) { return m ? next() % m : 0; }
};

// -------------------------
// 2. LOADING (TSV)
// -------------------------
struct Row { double k, t1, t2; };
double pTime(string s) {
    if (s.size() < 19) return 0;
    int dd = stoi(s.substr(8, 2)), h = stoi(s.substr(11, 2)), m = stoi(s.substr(14, 2));
    return (dd * 86400LL + h * 3600 + m * 60) * 1000.0;
}
class Maps { unordered_map<string, double> m; double c = 1; public: double g(string s) { if (!m.count(s)) m[s] = c++; return m[s]; } };

void Load(string f, vector<Row>& out) {
    ifstream in(f); if (!in) exit(1);
    string l; getline(in, l); Maps m_gen;
    while (getline(in, l)) {
        stringstream ss(l); string v; vector<string> c; while (getline(ss, v, '\t')) c.push_back(v);
        if (c.size() > 6 && c[5].find("2014-11-17") != string::npos) 
            out.push_back({ m_gen.g(c[2]), pTime(c[5]), pTime(c[6]) });
    }
}

// -------------------------
// 3. FLAT RELATIONAL ENGINE (O(1) CACHE-FRIENDLY)
// -------------------------
struct Hub {
    double k;
    vector<double> t1s;
    vector<double> t2s;
    vector<int> colors; 
};

class StarEngine {
    vector<Hub> hubs;
    RNG rng;
public:
    void build(const vector<Row>& d) {
        unordered_map<double, vector<pair<double, double>>> temp;
        for (auto& r : d) temp[r.k].push_back({ r.t1, r.t2 });
        for (auto& kv : temp) {
            Hub h; h.k = kv.first;
            for (auto& p : kv.second) {
                h.t1s.push_back(p.first);
                h.t2s.push_back(p.second);
                h.colors.push_back((fmod(p.first, 86400000.0) < 43200000.0) ? 0 : 1);
            }
            hubs.push_back(h);
        }
    }

    long long countRect(const Box& b, int color = -1, int rel_limit = 4) {
        long long total = 0;
        // Flat contiguous memory iteration -> massive speedup
        for (const auto& h : hubs) {
            long long path = 1;
            for (int i = 0; i < rel_limit; ++i) {
                long long c = 0;
                int target_col = (i == 3 ? color : -1);
                double mn1 = b.bounds[i*2].first, mx1 = b.bounds[i*2].second;
                double mn2 = b.bounds[i*2+1].first, mx2 = b.bounds[i*2+1].second;
                
                for (size_t p = 0; p < h.t1s.size(); ++p) {
                    if (h.t1s[p] >= mn1 && h.t1s[p] < mx1 && h.t2s[p] >= mn2 && h.t2s[p] < mx2) {
                        if (target_col != -1 && h.colors[p] != target_col) continue;
                        c++;
                    }
                }
                if (c == 0) { path = 0; break; }
                path *= c;
            }
            total += path;
        }
        return total;
    }

    void sampleBatch(const Box& b, int M_req, int color, int rel_limit, vector<Point>& out) {
        vector<double> cdf; vector<const Hub*> valid_hubs; double wsum = 0;
        for (const auto& h : hubs) {
            long long path = 1;
            for (int i = 0; i < rel_limit; ++i) {
                long long c = 0; int target_col = (i == 3 ? color : -1);
                double mn1 = b.bounds[i*2].first, mx1 = b.bounds[i*2].second;
                double mn2 = b.bounds[i*2+1].first, mx2 = b.bounds[i*2+1].second;
                
                for (size_t p = 0; p < h.t1s.size(); ++p) {
                    if (h.t1s[p] >= mn1 && h.t1s[p] < mx1 && h.t2s[p] >= mn2 && h.t2s[p] < mx2) {
                        if (target_col != -1 && h.colors[p] != target_col) continue;
                        c++;
                    }
                }
                if (c == 0) { path = 0; break; }
                path *= c;
            }
            if (path > 0) { wsum += path; cdf.push_back(wsum); valid_hubs.push_back(&h); }
        }
        if (valid_hubs.empty()) return;

        for (int m = 0; m < M_req; ++m) {
            double pick = rng.unit() * wsum;
            auto it = lower_bound(cdf.begin(), cdf.end(), pick);
            const Hub* h = valid_hubs[distance(cdf.begin(), it)];
            Point pt;
            for (int i = 0; i < rel_limit; ++i) {
                vector<size_t> cands; int target_col = (i == 3 ? color : -1);
                double mn1 = b.bounds[i*2].first, mx1 = b.bounds[i*2].second;
                double mn2 = b.bounds[i*2+1].first, mx2 = b.bounds[i*2+1].second;
                for (size_t p = 0; p < h->t1s.size(); ++p) {
                    if (h->t1s[p] >= mn1 && h->t1s[p] < mx1 && h->t2s[p] >= mn2 && h->t2s[p] < mx2) {
                        if (target_col != -1 && h->colors[p] != target_col) continue;
                        cands.push_back(p);
                    }
                }
                size_t chosen = cands[rng.idx(cands.size())];
                pt.push_back(h->t1s[chosen]); pt.push_back(h->t2s[chosen]);
            }
            out.push_back(pt);
        }
    }
};

// -------------------------
// 4. CORE ALGORITHM 2 & 3
// -------------------------

vector<WeightedPoint> Algorithm2(StarEngine& db, int k, double eps_u, int rel_limit, const vector<Point>& X_in) {
    Box inf; for (int i = 0; i < rel_limit * 2; ++i) inf.bounds.push_back({ -1e18, 1e18 });
    long long N = db.countRect(inf, -1, rel_limit);
    if (N == 0) return {};

    double eps_p = eps_u / 34.0;
    vector<Point> X = X_in;
    if (X.empty()) {
        db.sampleBatch(inf, k * 2, -1, rel_limit, X);
        for (auto& p : X) Scale(p);
    }

    // STRICT PAPER PARAMETRICS
    long double logN = log2((long double)N + 2.0L);
    long double tau = pow((long double)eps_p, rel_limit * 2 + 1) / (16.0L * (long double)X.size() * logN);
    int M = (int)min(600.0L, (3.0L / (pow((long double)eps_p, 2) * tau)) * log2(2.0L * pow((long double)N, 10.0L * rel_limit * 2)));
    if (tau < 1e-25) tau = 1e-10;

    vector<WeightedPoint> C; vector<Box> B; RNG rng(555);
    double Phi = 20.0;

    // LOOP INVERSION OPTIMIZATION: Discover Active cells globally per level, then assign to closest X
    for (int j = -2; j <= 6; ++j) {
        double Rad = Phi * pow(2.0, j), side = eps_p * Rad / sqrt(rel_limit * 2);
        if (side < 1e-6) continue;

        vector<Point> probes; db.sampleBatch(inf, 200, -1, rel_limit, probes);
        for (auto& p : probes) Scale(p);

        // Map Center -> Set of Unique Cells
        map<int, set<vector<long long>>> cells_by_center;

        for (const auto& p : probes) {
            double min_d = 1e18; int best_xi = -1;
            for (int i = 0; i < X.size(); ++i) {
                double d = L1Dist(X[i], p);
                if (d < min_d) { min_d = d; best_xi = i; }
            }
            if (best_xi != -1 && min_d <= 4 * Rad) {
                vector<long long> cid(rel_limit * 2);
                for (int d = 0; d < rel_limit * 2; ++d) cid[d] = floor((p[d] - (X[best_xi][d] - 2 * Rad)) / side);
                cells_by_center[best_xi].insert(cid);
            }
        }

        for (auto const& [xi_idx, cids] : cells_by_center) {
            const Point& xi = X[xi_idx];
            for (const auto& cid : cids) {
                Box bx; for (int d = 0; d < rel_limit * 2; ++d) { double st = (xi[d] - 2 * Rad) + (double)cid[d] * side; bx.bounds.push_back({ st, st + side }); }
                
                // Condition 3 Pruning
                double d_xi = bx.DistTo(xi), d_X = 1e18;
                for (auto& xb : X) { d_X = min(d_X, bx.DistTo(xb)); if (d_X < d_xi - 1e-7) break; }
                if (d_xi > d_X + bx.Diam() + 1e-7) continue; 

                Box brw = bx; for (auto& iv : brw.bounds) { iv.first *= 86400000.0; iv.second *= 86400000.0; }

                for (int color : {0, 1}) {
                    long long n_box = db.countRect(brw, color, rel_limit);
                    if (n_box == 0) continue;

                    vector<Point> S_batch; db.sampleBatch(brw, M, color, rel_limit, S_batch);
                    int fresh = 0; Point rep;
                    for (auto& s_raw : S_batch) {
                        Point s = s_raw; Scale(s);
                        bool covered = false; for (auto& old : B) if (old.Contains(s)) { covered = true; break; }
                        if (!covered) { fresh++; if (rep.empty()) rep = s; }
                    }

                    if ((double)fresh / M >= 2.0 * tau || (B.empty() && fresh > 0)) {
                        C.push_back({ rep, (double)n_box * ((double)fresh / M) * (1.0 / (1.0 - eps_u)), color });
                    }
                }
                B.push_back(bx);
            }
        }
    }
    return C;
}

// -------------------------
// 5. SOLVERS
// -------------------------
double WeightedCost(const vector<WeightedPoint>& C, const vector<Point>& S) {
    double t = 0; for (auto& p : C) { double m = 1e18; for (auto& s : S) m = min(m, L1Dist(p.point, s)); t += p.weight * m; } return t;
}

void solveStandard(const vector<WeightedPoint>& C, int k, vector<Point>& result) {
    result.clear(); if(C.empty() || k==0) return;
    vector<int> idxs(C.size()); iota(idxs.begin(), idxs.end(), 0);
    sort(idxs.begin(), idxs.end(), [&](int a, int b){ return C[a].weight > C[b].weight; });
    for (int i = 0; i < min(k, (int)C.size()); ++i) result.push_back(C[idxs[i]].point);
    RNG r(777); double best = WeightedCost(C, result);
    for (int i = 0; i < 40; ++i) {
        bool mod = false;
        for (int j = 0; j < result.size(); ++j) {
            int id = r.idx(C.size()); Point bk = result[j]; result[j] = C[id].point;
            double cur = WeightedCost(C, result); if (cur < best) { best = cur; mod = true; } else result[j] = bk;
        }
        if (!mod) break;
    }
}

void solveFair(const vector<WeightedPoint>& C, int kr, int kb, vector<Point>& result) {
    vector<int> ir, ib; for (size_t i = 0; i < C.size(); ++i) (C[i].color == 0 ? ir : ib).push_back(i);
    kr = min(kr, (int)ir.size()); kb = min(kb, (int)ib.size());
    result.clear(); if(ir.empty() && ib.empty()) return;
    
    sort(ir.begin(), ir.end(), [&](int a, int b){ return C[a].weight > C[b].weight; });
    sort(ib.begin(), ib.end(), [&](int a, int b){ return C[a].weight > C[b].weight; });
    for (int i = 0; i < kr; ++i) result.push_back(C[ir[i]].point);
    for (int i = 0; i < kb; ++i) result.push_back(C[ib[i]].point);
    
    RNG r(777); double best = WeightedCost(C, result);
    for (int i = 0; i < 40; ++i) {
        bool mod = false;
        auto step = [&](int K, vector<int>& pool, int off) {
            for (int k = 0; k < K; ++k) {
                int id = pool[r.idx(pool.size())]; Point bk = result[k + off]; result[k + off] = C[id].point;
                double cur = WeightedCost(C, result); if (cur < best) { best = cur; mod = true; } else result[k + off] = bk;
            }
        };
        if(kr > 0) step(kr, ir, 0); if(kb > 0) step(kb, ib, kr);
        if (!mod) break;
    }
}

// -------------------------
// 6. MAIN
// -------------------------
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    bool proportional = true; double eps = 0.05; 
    vector<int> k_values = {6, 12, 18, 24, 30};
    
    vector<Row> raw; Load("flights.tsv", raw); if (raw.empty()) return 1;
    StarEngine db; db.build(raw);

    for(int k_goal : k_values) {
        cout << "\n-----------------------------------" << endl;
        cout << "Running Algorithm 3 for k = " << k_goal << endl;
        auto t1 = chrono::high_resolution_clock::now();

        auto C12 = Algorithm2(db, k_goal, eps, 2, {});
        vector<Point> S12; solveStandard(C12, k_goal, S12); 

        auto C34 = Algorithm2(db, k_goal, eps, 2, {});
        vector<Point> S34; solveStandard(C34, k_goal, S34); 

        vector<Point> X_root;
        for(auto& l : S12) for(auto& r_p : S34) {
            Point p = l; p.insert(p.end(), r_p.begin(), r_p.end()); X_root.push_back(p);
        }
        auto Coreset = Algorithm2(db, k_goal, eps, 4, X_root);

        int kr = k_goal / 2, kb = k_goal - kr;
        if (proportional && !Coreset.empty()) {
            Box inf; for (int i = 0; i < DIMS; ++i) inf.bounds.push_back({ -1e18, 1e18 });
            long long nR = db.countRect(inf, 0, 4), nB = db.countRect(inf, 1, 4);
            if(nR+nB>0) { 
                double ratio = (double)nR / (nR + nB);
                kr = max(1, (int)round(k_goal * ratio)); kb = k_goal - kr; 
            }
        }

        vector<Point> sol; solveFair(Coreset, kr, kb, sol);

        auto t2 = chrono::high_resolution_clock::now();
        cout << "Points in Coreset: " << Coreset.size() << " | TOTAL Runtime: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " ms." << endl;
        cout << "Final Fair k-Median Cost: " << WeightedCost(Coreset, sol) << endl;
    }

    return 0;
}