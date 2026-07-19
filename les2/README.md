# İthaki Bilimkurgu Klasikleri: Veri Kazıma, Tokenizer & Fine-Tuning

> ⚠️ **Not:** Bu proje tamamen **eğitim amaçlı** hazırlanmıştır.

Bu proje, İthaki Yayın Grubu'nun **Bilimkurgu Klasikleri** serisine ait verilerin web ortamından kazınması, Hugging Face Hub üzerine yüklenmesi, özel bir Byte-Level BPE tokenizer eğitilmesi ve **Qwen3-4B-Instruct** modelinin Unsloth + LoRA ile fine-tune edilmesini kapsayan uçtan uca bir yapay zeka/veri işleme boru hattıdır.

---

## 📁 Proje Yapısı

| Dosya / Dizin | Açıklama |
|---|---|
| `ithaki_crawler.py` | `crawl4ai` ve `BeautifulSoup` kullanarak İthaki sitesinden kitap katalog verilerini (yazar, çevirmen, özet, fiyat vb.) asenkron olarak kazır. |
| `upload_dataset_to_hf.py` | Oluşturulan CSV kataloğunu ve README model kartını `gururaser/ithaki-bilimkurgu-klasikleri` Hugging Face Dataset deposuna yükler. |
| `train_tokenizer.py` | Veri setindeki metinlerden Hugging Face `tokenizers` kütüphanesi ile 1000 kelimelik Byte-Level BPE tokenizer eğitir ve HF Hub'a yükler. |
| `ithaki_qwen3_finetune.ipynb` | Veri setinden sentetik ChatML QA çiftleri üreterek Google Colab T4 GPU üzerinde `unsloth/Qwen3-4B-Instruct-2507` modelini LoRA ile fine-tune eder. |
| `requirements.txt` | Temel bağımlılıklar (`crawl4ai`, `beautifulsoup4`, `huggingface_hub`). |
| `ithaki_bilimkurgu_klasikleri_ozetli.csv` | Kazınmış ve işlenmiş kitap kataloğu verisi. |
| `bilimkurgu_bpe_tokenizer.json` | Eğitilmiş BPE tokenizer dosyasının yerel çıktısı. |

---

## 🚀 Kurulum

Gerekli Python paketlerini yükleyin:

```bash
pip install -r requirements.txt
```

---

## 🔄 Çalıştırma Adımları

1. **Veri Kazıma (Web Scrape):**
   ```bash
   python ithaki_crawler.py
   ```
2. **Hugging Face'e Dataset Yükleme:**
   ```bash
   python upload_dataset_to_hf.py
   ```
3. **Byte-Level BPE Tokenizer Eğitimi:**
   ```bash
   python train_tokenizer.py
   ```
4. **Model Fine-Tuning:**
   `ithaki_qwen3_finetune.ipynb` defterini Google Colab veya T4 GPU destekli bir ortamda çalıştırın.
