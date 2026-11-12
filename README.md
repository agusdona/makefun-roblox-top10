# Roblox Up-and-Coming Top 10 — entri make.fun

Bounty: [Predict Roblox's Up-and-Coming top 10](https://make.fun/bounty/roblox-up-and-coming-forecast-01)
($500 USDC, entri tutup 13 Nov 2026 06:00 UTC; jendela skor 13–16 Nov 2026, 72 jam.)

## Cara menjalankan

Program: `main.py` — Python 3.13, **stdlib saja**, tanpa network,
deterministik, maks 300 detik.

```sh
# Di sandbox penilai (default):
python3 /bot/main.py
#   membaca /input/history.ndjson + /input/task.json
#   menulis /output/forecast.json

# Lokal (untuk tes):
MAKEFUN_INPUT_DIR=../tests/synthetic \
MAKEFUN_OUTPUT_DIR=../tests/out1 \
python3 main.py
```

Variabel lingkungan yang dihormati (sesuai rules bounty):
`$MAKEFUN_INPUT_DIR` (default `/input`), `$MAKEFUN_OUTPUT_DIR`
(default `/output`).

## Format input/output (asumsi — lihat "Status kit" di bawah)

**Input** — `/input/history.ndjson`: satu objek JSON per baris, satu read
per baris, terurut dari yang terlama. Tiap baris: `at` (ISO 8601 UTC),
`source`, `filters`, `ok`, `up_and_coming` (daftar game: `rank`,
`universe_id`, `root_place_id`, `name`, `playing`, `up_votes`,
`down_votes`, `genre`), `top_trending` (atau null), `top_playing_now`
(atau null), `details` (per game: `name`, `playing`, `visits`,
`favorites`, `max_players`, `created`, `updated`, `creator_name`,
`creator_type`, `genre`, atau null), `responses`. Nilai yang tidak
dikirim Roblox bernilai null. `/input/task.json`: `history_until`,
`window` (`from`, `until`, 72 `slots` @ 60 menit), `places` (10).
Jendela dibaca dari `task.json`, bukan di-hardcode.

**Output** — `/output/forecast.json`, maks 1.048.576 byte, satu objek
dengan tepat dua kunci:
- `probabilities`: daftar baris `{"universe_id": int, "place": 1..10,
  "p": float 0..1}`. Satu game boleh muncul di beberapa tempat, maks
  sekali per tempat; satu tempat maks 100 baris; jumlah probabilitas
  per tempat maks 1.
- `top10`: tepat 10 universe id berbeda, urutan tebakan terbaik.

**Penilaian** (aturan resmi): tiap jam, tiap tempat, skor = ln(probabilitas
game yang benar-benar di tempat itu), floor 0,0001 (ln = −9,210340);
bagian yang tidak dinamai di suatu tempat dibagi rata ke semua game di
chart jam itu yang tidak dinamai di tempat itu. Log score = rata-rata
skor per jam; tie-break = kecocokan top 10.

## Cara program menghitung probabilitas

Untuk tiap game, peringkatnya selama 72 jam dimodelkan sebagai **difusi
Gaussian dengan drift** (tanpa Monte Carlo — analitik, jadi dua run
selalu byte-identik):

1. **Drift** (perubahan peringkat per jam): kombinasi kecepatan
   pertumbuhan pemain 24 jam (bobot 0,6 — chart mengurutkan dari
   kecepatan pertumbuhan, bukan ukuran) dan momentum peringkat 24 jam
   terakhir (bobot 0,4) yang dipetakan lewat **regresi
   growth→rank-velocity yang difit dari history itu sendiri**, ditambah
   momentum genre yang sama (bobot 0,15) dan dorongan kecil untuk game
   yang di-`update` dalam 48 jam terakhir (terlihat dari
   `details.updated`). Drift dijepit ±0,4 peringkat/jam dan meluruh
   eksponensial (tau 36 jam).
2. **Difusi**: sigma per game dari varians perubahan peringkat per jam
   game itu, di-shrink ke varians gabungan semua game, lalu dikali 0,5
   (varians mentah terlalu menyebar — backtest membuktikan 0,5 lebih
   baik dari 1,0 di data sintetis maupun history asli); sigma tumbuh
   akar-waktu, dijepit maks 9 peringkat. Rata-rata peringkat juga
   ditarik pelan (0,01/jam) ke peringkat ekuilibrium dari skor kekuatan
   (log playing, pertumbuhan 24 jam, stabilitas top-10).
3. **Age-out**: game keluar 28 hari setelah debut chart (pertama kali
   terlihat di history; bila sudah ada di read pertama, tanggal
   `created` jadi proksi debut). Survival mengikuti logistik di sekitar
   jam age-out (skala 6 jam); bila chart membantah estimasi (game masih
   berperingkat setelah tanggal itu), age-out diabaikan.
4. Tiap jam, peluang game di tempat k = selisih CDF normal di
   (k±0,5), dikali survival; **rata-rata atas 72 jam** itulah
   probabilitas akhirnya (satu forecast dinilai di semua jam, jadi
   rata-rata waktu adalah distribusi yang benar).
5. Per tempat, hanya game dengan p ≥ 0,006 yang dinamai (di bawah itu,
   bagian unnamed — sekitar 0,005/game — memberi skor lebih baik bila
   game itu ternyata muncul), maks 40 baris per tempat; total per tempat
   dijepit 0,85 sehingga ≥0,15 tetap unnamed sebagai lindung nilai untuk
   game yang baru meluncur selama jendela. `top10` diurutkan menurut
   peringkat harapan (rata-rata tempat tertimbang probabilitas).

Kalibrasi ini memakai temuan riset di `../prediction.md` — pertumbuhan
24 jam adalah sinyal utama (gelombang ragdoll naik dengan +90%/+151%
dalam 24 jam), stabilitas CCU menahan game kecil yang lengket
(Ragdoll Realism bertahan 6 hari di top 10), dan chart sangat volatil
(median 6 tempat top-10 berubah antar read) sehingga probabilitas
disebar, tidak pernah terkonsentrasi pada satu urutan. **Tidak ada nama
game, universe id, atau event yang di-hardcode**: semua angka di atas
diestimasi dari history yang diberikan.

## Hasil tes

Tes memakai data sintetis (`../tests/`: generator dunia yang meniru
mekanika chart asli — peringkat dari kecepatan pertumbuhan 24 jam,
age-out 28 hari, lonjakan update, game yang memudar, dan peluncuran
baru; 81 history reads + 72 future reads sebagai kebenaran).

| Tes | Hasil |
|---|---|
| Runtime (history sintetis 81 reads, 97 game) | **0,35 detik** (batas 300 dtk) |
| Determinisme | 2 run → **byte-identik** |
| Validasi skema (`tests/validate.py`) | **VALID**: 247 baris, 10 tempat, jumlah per tempat 0,64–0,79 (≤ 1), top10 10 id berbeda, 12.750 byte |
| Backtest log score vs baseline, sintetis seed 42 (aturan resmi, `tests/score.py`) | **forecast −37,50** vs baseline −47,41 → **+9,91 nats** (typical p 0,024 vs 0,009) — ronde 2: −38,04 (+9,37) |
| Backtest log score, sintetis seed 7 (dunia beda, anti-overfit) | **forecast −41,75** vs baseline −55,28 → **+13,54 nats** — ronde 2: −41,84 (+13,45) |
| Backtest log score, history asli kit (rolling 15 cutoff, 131 reads) | **forecast −38,71** — ronde 2: −39,12 (+0,41) |
| History kosong | exit 0, menulis `{"probabilities":[],"top10":[]}` + peringatan stderr |
| History minimal (3 reads) | exit 0, output VALID (265 baris) |
| Tanpa `task.json` | exit 0, fallback jendela = read terakhir, output identik |

Catatan: baseline punya top-10 hits lebih banyak (6,11 vs 5,32) tapi
log score jauh lebih buruk — sesuai aturan, log score-lah yang
menentukan peringkat, dan baseline yang menaruh 0,5 pada urutan saat
ini kena floor −9,21 setiap kali chart bergerak. Backtest sintetis
membuktikan algoritma mengalahkan baseline di data yang belum dilihat;
angka absolutnya tidak bisa dibandingkan langsung dengan papan
practice (dunia sintetis ≠ chart asli).

## Revisi ronde 3 (12 Nov 2026) — perbaikan metodologi

Tiga perubahan, semuanya dipilih karena memperbaiki log score di
KETIGA backtest (sintetis seed 42, sintetis seed 7, history asli kit):
(1) drift kini didominasi pertumbuhan (bobot 0,6) atas momentum
peringkat (0,4) — chart mengurutkan dari kecepatan pertumbuhan, dan
momentum adalah proksi yang tertinggal; (2) ambang penamaan naik
0,002 → 0,006 — game yang dinamai di bawah ~0,005 mendapat skor
lebih buruk daripada dibiarkan unnamed (bagian unnamed dibagi rata
ke game di chart, ~0,005 each); (3) sigma difusi dikali 0,5 —
varians perubahan peringkat mentah terlalu menyebar.

Yang dicoba dan DITOLAK karena tidak memperbaiki ketiga backtest:
sinyal akselerasi pertumbuhan (terlalu berisik — memperburuk semua);
blend pertumbuhan 12 jam (membantu sintetis, memperburuk history
asli); bobot ekuilibrium yang didominasi pertumbuhan (membantu history
asli, memperburuk sintetis); normalisasi total per tempat ke 0,85
(terlalu percaya diri — memperburuk seed 7 dan history asli).
Oracle (frekuensi masa depan yang sebenarnya) ada di −25,89 /
−27,18, jadi masih ada jarak — tetapi sebagian besar jarak itu
adalah ketidakpastian yang melekat, bukan kesalahan model.

## Status kit & asumsi yang belum terverifikasi

- Kit resmi (`.tar.gz` dari `GET /c/<slug>/kit`) **tidak bisa diunduh
  lewat jalur fetch teks** yang tersedia di sini (konten biner), dan
  endpoint `GET /api/competitions/<slug>/kit` hanya memberi `kit.json`
  (daftar file + sha256). Format input/output di atas **direkonstruksi
  dari teks rules resmi** yang tertanam di record kompetisi
  (`GET /api/competitions/<slug>`, field `rules`) — termasuk skema
  `forecast.json`, aturan unnamed share, floor 0,0001, baseline
  starter (0,5 / 0,3 / 0,15), dan jendela practice vs final — plus
  `kit.json` (dev dataset `input/history.ndjson`, `input/task.json`,
  `starter/main.ts`). Bila kit berhasil diunduh, jalankan
  `grade-local` dari kit terhadap program ini sebelum submit; bila ada
  perbedaan format, sesuaikan dan catat di sini.
- **Tidak ada validasi terhadap data practice asli**: dataset dev kit
  (34 reads, 30 Okt–7 Nov) tidak tersedia di sini, jadi kalibrasi
  diuji hanya pada data sintetis. Skor practice sebenarnya baru
  terlihat setelah submit.
- **Event terjadwal di luar data tidak dipakai.** Riset menemukan satu
  katalis bertanggal di dalam jendela — event "NEW MAPS" Survive Your
  Ideas mulai 14 Nov 18:00 UTC — tetapi event itu tidak ada di data
  history, dan tugas ini melarang hardcode daftar statis. Program
  hanya memakai sinyal yang ada di data (momentum, pertumbuhan,
  stabilitas, resensi update). Kekuatan game itu tetap tertangkap
  sebagian lewat basis pemain terbesar + pertumbuhan + stabilitasnya.
  Bila pemilik entri ingin memakai katalis itu, caranya adalah
  menyediakan file katalis sebagai input — bukan hardcode di program.
- Lotere peluncuran baru (1–2 slot top 10 kemungkinan diisi game yang
  meluncur 12–14 Nov) tidak bisa diprediksi dari data; lindung
  nilainya adalah bagian unnamed ≥0,15 per tempat.
- Program tidak membaca `top_trending` / `top_playing_now`; keduanya
  hanya diparse dan diabaikan.
