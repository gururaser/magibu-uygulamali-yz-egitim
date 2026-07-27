# 🛒 E-Commerce Synthetic NER Dataset Generator with NVIDIA NeMo Data Designer

Bu proje, **NVIDIA NeMo Data Designer** (`data-designer`) ve **Hugging Face Inference Provider** (`deepseek-ai/DeepSeek-V4-Flash:fireworks-ai`) altyapısını kullanarak 1.700+ adet doğal Türkçe e-ticaret ürün adı ve bunlara ait Named Entity Recognition (NER) etiketlerini içeren sentetik veri üretmek üzere geliştirilmiştir.

---

## 📊 Üretim Metrikleri & Maliyet Özeti

- **Üretilen Canlı Veri Sayısı**: 1.700 Adet (17 Batch x 100 Adet)
- **Eşzamanlı İstek Kapasitesi (`max_parallel_requests`)**: 20-50 İstek / An
- **Model**: `deepseek-ai/DeepSeek-V4-Flash:fireworks-ai` (Fireworks AI Router)
- **Sağlayıcı**: Hugging Face Inference Provider
- **Sağlayıcı Tarifesi**: 1M Girdi Token = **$0.14 USD** | 1M Çıktı Token = **$0.28 USD**
- **Reasoning (Düşünme Token'ları)**: 0 (Devre dışı bırakıldı - `--no-reasoning`)
- **Toplam Girdi Token (Input Tokens)**: **~790.500 Token** (~791K)
- **Toplam Çıktı Token (Output Tokens)**: **~232.900 Token** (~233K)
- **Toplam Token Harcaması**: **~1.023.400 Token** (~1.02M)
- **Saf Token Maliyeti (Hesaplanan Net LLM)**: **~$0.18 USD** (1.700 Adet Veri İçin)
- **Gerçekleşen HF Panel Harcaması**: **$0.84 USD** (Ağ zaman aşımları, retries ve sağlık kontrolleri dahil)
- **Çıktı Formatı**: JSON Lines (`.jsonl`)

---

## 📁 Proje Yapısı

```
les4/nemo_datadesigner_ecommerce_ner/
├── generate_ecommerce_ner.py   # NeMo Data Designer + HF Router sentetik veri üreticisi
├── push_to_hub.py              # Üretilen veriyi HF Hub'a yükleyen ve kart oluşturan betik
├── sample_seeds.json           # 10 özgün e-ticaret kategorisi için taksonomi seed verisi
├── requirements.txt            # Proje bağımlılıkları (data-designer, huggingface_hub vb.)
├── .env.example                # Örnek HF_TOKEN ortam değişkeni dosyası
├── .env                        # Gerçek HF_TOKEN ortam değişkeniniz
├── README.md                   # Proje ve canlı terminal kayıt belgesi
└── data/
    └── ecommerce_ner_dataset.jsonl # Üretilen sentetik veri seti (.jsonl)
```

---

## 🏷️ NER Etiket Standardı (8 Anahtar)

1. **`BRAND`**: Marka (*Nike, Samsung, Enza Home, Pierre Cardin*)
2. **`CATEGORY`**: Ürün Türü / Kategorisi (*Spor Ayakkabı, Masa Örtüsü, Akıllı Telefon*)
3. **`MODEL`**: Model / Ürün Serisi (*Air Max 270, Viyana, Galaxy S24*)
4. **`COLOR`**: Renk (*Siyah, Bej, Uzay Grisi, Lacivert*)
5. **`SIZE_VARIANT`**: Beden / Boyut / Kapasite (*42 Numara, 150x220 cm, XL, 128 GB*)
6. **`GENDER_TARGET`**: Hedef Kitle (*Erkek, Kadın, Çocuk, Unisex*)
7. **`MATERIAL`**: Malzeme / Kumaş Bileşimi (*Pamuk, Deri, Paslanmaz Çelik*)
8. **`SPECIFICATION`**: Nitelik / Özellik / Stil (*Kareli, Kablosuz, Su Geçirmez, Mat Bitiş*)

---

## 📝 Örnek Üretilen Veri Çıktısı (JSON Lines)

```json
{"id": "ecom_ner_00001", "product_name": "İstikbal Nova Koltuk Takımı Gri Kumaş 3+1+1", "category_domain": "Mobilya & Dekorasyon", "entities": [{"text": "İstikbal", "label": "BRAND", "start": 0, "end": 8}, {"text": "Nova", "label": "MODEL", "start": 9, "end": 13}, {"text": "Koltuk Takımı", "label": "CATEGORY", "start": 14, "end": 27}, {"text": "Gri", "label": "COLOR", "start": 28, "end": 31}, {"text": "Kumaş", "label": "MATERIAL", "start": 32, "end": 37}, {"text": "3+1+1", "label": "SIZE_VARIANT", "start": 38, "end": 43}]}
{"id": "ecom_ner_00002", "product_name": "Adidas Ultraboost 22 Beyaz Kadın Koşu Ayakkabısı 39", "category_domain": "Ayakkabı", "entities": [{"text": "Adidas", "label": "BRAND", "start": 0, "end": 6}, {"text": "Ultraboost 22", "label": "MODEL", "start": 7, "end": 20}, {"text": "Beyaz", "label": "COLOR", "start": 21, "end": 26}, {"text": "Kadın", "label": "GENDER_TARGET", "start": 27, "end": 32}, {"text": "Koşu Ayakkabısı", "label": "CATEGORY", "start": 33, "end": 48}, {"text": "39", "label": "SIZE_VARIANT", "start": 49, "end": 51}]}
{"id": "ecom_ner_01544", "product_name": "Samsung 65QN90C 65'' 4K Smart QLED TV Siyah", "category_domain": "Elektronik", "entities": [{"text": "Samsung", "label": "BRAND", "start": 0, "end": 7}, {"text": "65QN90C", "label": "MODEL", "start": 8, "end": 15}, {"text": "65''", "label": "SIZE_VARIANT", "start": 16, "end": 20}, {"text": "4K Smart QLED TV", "label": "CATEGORY", "start": 21, "end": 37}, {"text": "Siyah", "label": "COLOR", "start": 38, "end": 43}]}
```

---

## 💻 Kullanım Komutları

Ana proje dizininde sanal ortamı aktif ettikten sonra aşağıdaki komutları kullanabilirsiniz:

### 1. Hızlı Canlı LLM Üretimi (Otomatik Retry + Bekleme Mekanizmalı)
```bash
source .venv/bin/activate
python3 les4/nemo_datadesigner_ecommerce_ner/generate_ecommerce_ner.py --count 1000 --batch-size 100 --max-parallel 25 --no-reasoning --max-retries 5 --retry-delay 3.0
```

> **Hata Toleransı (Exponential Backoff)**: Bağlantı kesilmesi veya zaman aşımında kod çökmez; 3s, 6s, 12s, 24s gibi artan aralıklarla 5 defa otomatik tekrar dener (`--max-retries 5`).
> **Not**: Kod varsayılan olarak mevcut `.jsonl` dosyasındaki verileri korur ve yeni verileri id sırasını bozmadan dosyanın sonuna ekler. Sıfırdan temiz bir dosya başlatmak isterseniz komutun sonuna `--overwrite` ekleyin.

### 2. Önizleme Testi (5 Adet Canlı LLM Üretimi)
```bash
source .venv/bin/activate
python3 les4/nemo_datadesigner_ecommerce_ner/generate_ecommerce_ner.py --preview --no-reasoning
```

### 3. Çevrimdışı (Offline) Seed Test Üretimi
```bash
source .venv/bin/activate
python3 les4/nemo_datadesigner_ecommerce_ner/generate_ecommerce_ner.py --offline --count 100
```

### 4. Hugging Face Hub'a Yükleme
```bash
source .venv/bin/activate
python3 les4/nemo_datadesigner_ecommerce_ner/push_to_hub.py --repo-id "kullanici_adiniz/turkish-ecommerce-ner-dataset"
```
