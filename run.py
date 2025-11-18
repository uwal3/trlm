import torch
from transformers import AutoTokenizer

from model.config import TRLMConfig
from model.trlm import TRLM, TRLMCarry
from model.loss_head import ACTLossHead


@torch.no_grad()  # <--- КРИТИЧЕСКИ ВАЖНО для инференса
def generate(model, tokenizer, prompt, max_new_tokens, temperature=1, top_k=50):
    """
    Функция для генерации текста с помощью модели TRLM.
    """
    model.eval()  # <--- Переводим модель в режим оценки

    # 1. Токенизируем входной промпт
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    # 2. Цикл генерации, токен за токеном
    for _ in range(max_new_tokens):
        # --- Обрезка контекста ---
        # Если последовательность становится слишком длинной, обрезаем ее слева
        if input_ids.size(1) > model.model.config.block_size:  # type: ignore
            input_ids = input_ids[:, -model.model.config.block_size :]  # type: ignore

        # --- Forward pass ---
        # Для простого инференса нам не нужно управлять carry между шагами.
        # Модель сама создаст начальный carry из input_ids.
        # Это не самый быстрый способ (KV-кэш был бы лучше), но самый простой и надежный.
        carry = model.initial_carry(
            {"input_ids": input_ids}, device=device
        )  # target тут не важен
        _, outputs = model.model(carry=carry, new_samples={"input_ids": input_ids})

        # 3. Получаем логиты для ПОСЛЕДНЕГО токена
        logits = outputs["logits"][:, -1, :]  # Shape: (batch_size, vocab_size)

        # 4. Сэмплирование (чтобы текст был нескучным)
        # a) Применяем температуру
        logits = logits / temperature

        # b) Top-K сэмплирование
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")  # Обрезаем все, что не в top-k

        # c) Получаем вероятности и сэмплируем следующий токен
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # 5. Добавляем новый токен к последовательности
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

        # 6. Проверяем, не сгенерирован ли токен конца последовательности
        if next_token_id == tokenizer.eos_token_id:
            break

    # 7. Декодируем все обратно в текст
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    model.train()  # Возвращаем модель в режим обучения на всякий случай
    return generated_text


# --- Главный блок для запуска ---
if __name__ == "__main__":
    # --- Загрузка модели ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "/Users/nikita/Downloads/zeta2/trlm-ckpt-40000.pt"  # УКАЖИ ПУТЬ К СВОЕМУ ЛУЧШЕМУ ЧЕКПОИНТУ

    print(f"Загрузка чекпоинта из {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["model_args"]

    gptconf = TRLMConfig(**model_args)
    base_model = TRLM(gptconf)
    model = ACTLossHead(base_model)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print("Модель загружена.")

    # --- Загрузка токенизатора ---
    # Используем стандартный токенизатор, например, gpt2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Важно для некоторых моделей

    # --- Генерация ---
    prompt = "one plus one always equals"

    print(f"\nПромпт:\n{prompt}\n")
    print("-" * 50)
    print("Генерация...\n")

    generated_output = generate(model, tokenizer, prompt, max_new_tokens=100)

    print(f"Результат:\n{generated_output}")
