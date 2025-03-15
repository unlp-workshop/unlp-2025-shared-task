from openai import AsyncOpenAI
from openai import OpenAI, AsyncOpenAI


class InferenceClient:
    def __init__(
        self,
        api_key,
        api_url,
        model="default_model",
        temperature=0.6,
        top_p=0.9,
        max_tokens=4096,
        continue_final_message=False,
        add_generation_prompt=True,
        async_client=False,
        prepend_to_generation="",
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.continue_final_message = continue_final_message
        self.add_generation_prompt = add_generation_prompt
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.api_url, timeout=2400)
            if not async_client
            else AsyncOpenAI(api_key=self.api_key, base_url=self.api_url, timeout=2400)
        )
        self.prepend_to_generation = prepend_to_generation

    async def _get_model_id(self):
        if self.model == "default_model":
            models = self.client.models.list()
            self.model = models.data[0].id
        return self.model

    async def generate_completion(self, messages):
        model_id = await self._get_model_id()
        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body={
                "contine_final_message": self.continue_final_message,
                "add_generation_prompt": self.add_generation_prompt,
            },
        )
        content = response.choices[0].message.content
        completion_tokens = response.usage.completion_tokens
        # TODO: retrieve num generated tokens
        return {
            "generations": self.prepend_to_generation + content,
            "generated_token_count": completion_tokens,
        }
