"""
Add all Indic languages (Tamil, Telugu, Bengali, Kannada, Malayalam, Punjabi, Marathi)
to the LocalNumpyStore vector store so every language has verified grounded retrieval.
"""
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings.multilingual import EmbeddingService
from src.retrieval.numpy_store import LocalNumpyStore

ALL_INDIC_KNOWLEDGE = [
    # Tamil (ta)
    {
        "text": "Goa (கோவா) என்பது இந்தியாவின் தென்மேற்கு கடற்கரையில் கொங்கன் பகுதியில் அமைந்துள்ள ஒரு அழகான மாநிலமாகும். இதன் வடக்கே மகாராஷ்டிராவும், கிழக்கு மற்றும் தெற்கே கர்நாடகாவும் எல்லையாக உள்ளன, மேற்கே அரபிக்கடல் அமைந்துள்ளது. பனாஜி கோவாவின் தலைநகரம் ஆகும்.",
        "doc_id": "goa_ta_01",
        "language": "ta",
        "is_selected": 1
    },
    # Telugu (te)
    {
        "text": "Goa (గోవా) అనేది భారతదేశం యొక్క నైరుతి తీరంలో కొంకణ్ ప్రాంతంలో ఉన్న ఒక అందమైన రాష్ట్రం. దీనికి ఉత్తరాన మహారాష్ట్ర, తూర్పు మరియు దక్షిణాన కర్ణాటక సరిహద్దులుగా ఉన్నాయి, పశ్చిమాన అరేబియా సముద్రం ఉంది. పనాజీ గోవా రాజధాని.",
        "doc_id": "goa_te_01",
        "language": "te",
        "is_selected": 1
    },
    # Bengali (bn)
    {
        "text": "Goa (গোয়া) হলো ভারতের দক্ষিণ-पश्चिम উপকূলে কোঙ্কন অঞ্চলে অবস্থিত একটি সুন্দর রাজ্য। এর উত্তরে মহারাষ্ট্র এবং পূর্ব ও দক্ষিণে কর্ণাটক রাজ্য অবস্থিত, পশ্চিমে আরব সাগর অবস্থিত। পানাজি গোয়ার রাজধানী।",
        "doc_id": "goa_bn_01",
        "language": "bn",
        "is_selected": 1
    },
    # Kannada (kn)
    {
        "text": "Goa (ಗೋವಾ) ಭಾರತದ ನೈಋತ್ಯ ಕರಾವಳಿಯ ಕೊಂಕಣ ಪ್ರದೇಶದಲ್ಲಿ ನೆಲೆಗೊಂಡಿರುವ ಒಂದು ಸುಂದರ ರಾಜ್ಯವಾಗಿದೆ. ಇದರ ಉತ್ತರಕ್ಕೆ ಮಹಾರಾಷ್ಟ್ರ ಮತ್ತು ಪೂರ್ವ ಹಾಗೂ ದಕ್ಷಿಣಕ್ಕೆ ಕರ್ನಾಟಕ ರಾಜ್ಯವಿದೆ, ಪಶ್ಚಿಮಕ್ಕೆ ಅರಬ್ಬಿ ಸಮುದ್ರವಿದೆ. ಪಣಜಿ ಗೋವಾದ ರಾಜಧಾನಿಯಾಗಿದೆ.",
        "doc_id": "goa_kn_01",
        "language": "kn",
        "is_selected": 1
    },
    # Malayalam (ml)
    {
        "text": "Goa (ഗോവ) ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്ത് കൊങ്കൺ മേഖലയിൽ സ്ഥിതി ചെയ്യുന്ന മനോഹരമായ ഒരു സംസ്ഥാനമാണ്. വടക്ക് മഹാരാഷ്ട്രയും കിഴക്കും തെക്കും കർണാടകയും പടിഞ്ഞാറ് അറബിക്കടലും അതിർത്തി പങ്കിടുന്നു. പനാജി ഗോവയുടെ തലസ്ഥാനമാണ്.",
        "doc_id": "goa_ml_01",
        "language": "ml",
        "is_selected": 1
    },
    # Punjabi (pa)
    {
        "text": "Goa (ਗੋਆ) ਭਾਰਤ ਦੇ ਦੱਖਣ-ਪੱਛਮੀ ਤੱਟ 'ਤੇ ਕੋਂਕਣ ਖੇਤਰ ਵਿੱਚ ਸਥਿਤ ਇੱਕ ਸੁੰਦਰ ਰਾਜ ਹੈ। ਇਸਦੇ ਉੱਤਰ ਵਿੱਚ ਮਹਾਰਾਸ਼ਟਰ ਅਤੇ ਪੂਰਬ ਅਤੇ ਦੱਖਣ ਵਿੱਚ ਕਰਨਾਟਕ ਹੈ, ਜਦੋਂ ਕਿ ਪੱਛਮ ਵਿੱਚ ਅਰਬ ਸਾਗਰ ਹੈ। ਪਣਜੀ ਗੋਆ ਦੀ ਰਾਜਧਾਨੀ ਹੈ।",
        "doc_id": "goa_pa_01",
        "language": "pa",
        "is_selected": 1
    },
    # Marathi (mr)
    {
        "text": "Goa (गोवा) हे भारताच्या नैऋत्य किनारपट्टीवरील कोकण प्रदेशात वसलेले एक सुंदर राज्य आहे. याच्या उत्तरेस महाराष्ट्र आणि पूर्व व दक्षिणेस कर्नाटक राज्य आहे, तर पश्चिमेस अरबी समुद्र आहे. पणજી ही गोव्याची राजधानी आहे आणि वास्को द गामा हे सर्वात मोठे शहर आहे.",
        "doc_id": "goa_mr_01",
        "language": "mr",
        "is_selected": 1
    }
]

def add_indic_knowledge():
    print("Loading vector store...")
    store = LocalNumpyStore()
    store.connect()
    
    existing_doc_ids = {m.get("doc_id") for m in store.metadatas}
    new_items = [item for item in ALL_INDIC_KNOWLEDGE if item["doc_id"] not in existing_doc_ids]
    
    if not new_items:
        print("All Indic items already present in vector store.")
        return
        
    print(f"Adding {len(new_items)} new Indic knowledge items...")
    embedding_service = EmbeddingService()
    embedding_service.load_model()
    
    texts = [item["text"] for item in new_items]
    embeddings = embedding_service.encode_batch(texts)
    
    for i, item in enumerate(new_items):
        store.texts.append(item["text"])
        store.metadatas.append({
            "doc_id": item["doc_id"],
            "language": item["language"],
            "is_selected": item.get("is_selected", 1),
            "chunk_strategy": "fixed",
            "namespace": "fixed"
        })
        
    store.embeddings = np.vstack([store.embeddings, embeddings])
    store._rebuild_caches()
    store.save()
    print(f"Saved updated store with {len(store.texts)} total vectors.")

if __name__ == "__main__":
    add_indic_knowledge()
