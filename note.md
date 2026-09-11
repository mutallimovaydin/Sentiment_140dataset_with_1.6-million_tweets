# Twitter NLP Pipeline & Zero-Shot Classification

Bu layihə Kaggle-ın "Sentiment140" məlumat bazasından istifadə edərək tvitlərin emalını, canlı axın simulyasiyasını və süni intellekt modelləri ilə mövzu təsnifatını nümayiş etdirir.

**Əsas Funksionallıqlar**
* **Məlumatların Yüklənməsi:** kaggle vasitəsilə `kazanova/sentiment140` datasetinin yüklənməsi və arxivdən çıxarılması.
* **Mətnin Təmizlənməsi (Text Cleaning):** Mətnlər tamamilə kiçik hərflərə çevrilir. URL-lər, istifadəçi etiketləri, HTML xüsusi simvolları və qeyri-əlifba işarələri `re` (regex) modulu vasitəsilə mətnlərdən təmizlənir.
* **Axın (Stream) Simulyasiyası:** Generatorlar (`yield`) və `time.sleep` funksiyası köməyi ilə tvitlərin müəyyən fasilələrlə (məsələn, 0.5 saniyə) canlı gəlməsi simulyasiya olunur.
* **Sıfır Atışlı Təsnifat (Zero-Shot Classification):** Hugging Face `transformers` kitabxanasının `facebook/bart-large-mnli` modeli istifadə olunur. Model vasitəsilə tvitlər əvvəlcədən təyin olunmuş kateqoriyalara ("Sports", "Politics", "Tech", "Entertainment") görə təsnif edilir və əminlik faizi göstərilir.

**İstifadə Olunan Texnologiyalar**
* Verilənlərin idarəsi: `pandas`, `numpy`
* Təbii Dil Emalı (NLP): `transformers` (pipeline modulu)
* Məlumatların miqyaslanması (Əlavə olaraq daxil edilib): `scikit-learn`
* Qrafik və Vizuallaşdırma: `seaborn`, `matplotlib`, `plotly.express`

**İşləmə Ardıcıllığı**
1. Dataset `pandas` vasitəsilə oxunur və sütun adları daha oxunaqlı formaya salınır (`target`, `ids`, `date`, `flag`, `user`, `text`).
2. Baza üzərindən 600,000 təsadüfi tvit seçilərək NLP üçün təmizlənir və yalnız 3 simvoldan uzun olan yekun mətnlər saxlanılır.
3. Təmizlənmiş verilənlər axın simulyatoruna ötürülür. Modelləşdirmə mərhələsində hər bir tvitin aid olduğu kateqoriya və modelin əminlik dərəcəsi hesablanaraq ekrana yazdırılır.
