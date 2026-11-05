from poseidon.train_language import encode_record

class Tokenizer:
    eos_token_id = 9
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        return [1] * len(messages[0]["content"])
    def encode(self, text, **kwargs):
        return [2] * len(text)

def test_only_assistant_tokens_are_supervised():
    record = {"id": "example", "messages": [{"role": "system", "content": "Never learn this instruction as an answer."}, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}, {"role": "user", "content": "second"}, {"role": "assistant", "content": "unrelated answer"}]}
    result = encode_record(Tokenizer(), record, max_length=64)
    assert result["labels"] == [-100]*5 + [2]*5 + [9]
    assert len(result["input_ids"]) == len(result["labels"])

def test_long_prompt_is_rejected_not_detached_from_answer():
    record = {"id": "long", "messages": [{"role":"user","content":"x"*100}, {"role":"assistant","content":"y"}]}
    assert encode_record(Tokenizer(), record, max_length=64) is None

def test_missing_answer_is_rejected():
    record = {"id":"missing", "messages":[{"role":"user","content":"hello"}]}
    assert encode_record(Tokenizer(), record) is None
