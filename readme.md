# Tabel Ringkasan Konsep AdaBoost

## 1. Konsep Dasar AdaBoost

| Konsep | Penjelasan |
|--------|------------|
| **Stump (Weak Learner)** | Model dasar berupa *decision stump* (pohon 1-level). Hanya membuat satu pertanyaan -> sangat sederhana tetapi efektif saat digabungkan |
| **Bobot Suara Berbeda (Amount of Say)** | Setiap stump diberi bobot berdasarkan akurasinya. Stump yg lebih akurat memiliki pengaruh lebih besar pada prediksi akhir |
| **Belajar dari Kesalahan** | Setiap stump dibuat untuk memperbaiki kesalahan stump sebelumnya. Sampel yg salah diklasifikasikan akan diberi bobot lebih tinggi |

## 2. Workflow / Alur Kerja AdaBoost

| Tahap | Penjelasan |
|-------|------------|
| **1. Inisialisasi Bobot Data** | Semua sampel diberi bobot yg sama. Tidak ada sampel yang lebih dominan |
| **2. Memilih Stump Terbaik** | Algoritma mencari fitur + threshold dengan kesalahan terkecil untuk membuat stump pertama |
| **3. Menghitung Amount of Say (α)** | Mengukur kontribusi stump berdasarkan tingkat kesalahannya. Stump yang akurat -> α besar |
| **4. Memperbarui Bobot Sampel** | Sampel yg salah -> bobotnya dinaikkan; sampel yg benar -> bobotnya diturunkan sehingga model fokus pada data sulit |
| **5. Iterasi / Pengulangan** | Proses diulang sehingga setiap stump baru memperbaiki kesalahan stump sebelumnya hingga model selesai |

## 3. Mekanisme Akhir AdaBoost

| Mekanisme | Penjelasan |
|-----------|------------|
| **Voting Terbobot** | Prediksi akhir dibuat dari gabungan stump dengan bobot suara masing-masing |
| **Weak -> Strong Learner** | Banyak model sederhana digabungkan menjadi satu model klasifikasi yang kuat |
| **Fokus pada Kesalahan** | Algoritma fokus pada data yang sulit diklasifikasikan dengan menaikkan bobotnya |
| **Model Berurutan** | Stump dilatih *secara sekuensial*, tidak paralel seperti di Random Forest |