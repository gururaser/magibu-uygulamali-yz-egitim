# Ödev 1 — Custom Chat Template

Bu klasör, diğer teslimattan bağımsız bir Transformers chat-template çalışmasıdır. HF Router sunucu şablonunu değiştirmez; `chat_template.jinja` yerel olarak test edilebilir ve terminalde render edilebilir.

## Kapsam

Şablon ChatML tarzı işaretlerle `system`, `user`, `assistant` ve `tool` rollerini destekler. Sistem mesajından sonra araç tanımları JSON olarak gösterilir; bir asistan mesajında birden fazla `tool_calls`, çağrı kimliği, fonksiyon adı/argümanları ve `tool` sonucu render edilir. Desteklenmeyen rol `raise_exception` ile reddedilir. `add_generation_prompt=True` açık asistan başlangıcı üretir.

## Kurulum ve doğrulama

Proje kök `.venv`siyle:

```bash
source .venv/bin/activate
pip install -r les6/odev1_custom_chat_template/requirements.txt
cd les6/odev1_custom_chat_template
python -m pytest -q -W error
python render_template.py
```

`template.py` içindeki `render_chat()` fonksiyonu, Transformers tokenizer indirmeden Jinja render testi sağlar. Proje kökünden modül olarak çalıştırmak için `PYTHONPATH=les6 python -m odev1_custom_chat_template.render_template` kullanılır.

## Örnek

```python
from odev1_custom_chat_template.template import render_chat

print(render_chat(
    [
        {"role": "system", "content": "Hava durumu hakkında yardımcı ol."},
        {"role": "user", "content": "İstanbul hava durumunu göster."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"İstanbul\"}"}}
        ]},
        {"role": "tool", "tool_call_id": "call-1", "name": "get_weather", "content": '{"city": "İstanbul", "temperature_c": 24}'},
    ],
    tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    add_generation_prompt=True,
))
```

Render çıktısı `<|im_start|>system`, `<|tool_call|>`, `<|tool_result|>` ve son `<|im_start|>assistant` işaretlerini içerir. Terminal demosu normal sohbeti, çağrıyı ve sonucu ayrı başlıklarla gösterir.

## Teslim

GitHub klasörü: <https://github.com/gururaser/magibu-uygulamali-yz-egitim/tree/main/les6/odev1_custom_chat_template>
