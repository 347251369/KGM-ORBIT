import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

class JSONBackend:

    def __init__(self,
        *,
        endpoint,
        model,
        api_key_env,
        temperature,
        max_tokens,
        timeout,
        cache_dir=None,
        replay=False,
        model_options=None):
        self.endpoint = endpoint
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.replay = replay
        self.model_options = dict(model_options or {})
        if set(self.model_options) & {'model', 'messages', 'temperature', 'max_tokens', 'response_format'}:
            raise ValueError('Model options cannot replace core request fields')

    def complete(self, prompt, payload):
        request = dict(model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[dict(role='system', content=prompt), dict(role='user', content=json.dumps(payload, ensure_ascii=False))],
            response_format={'type': 'json_object'})
        request.update(self.model_options)
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f'{key}.json' if self.cache_dir else None
        if self.replay:
            if path is None:
                raise ValueError('Replay requires a cache directory')
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['request_hash'] != key:
                raise ValueError('Replay request mismatch')
            response = record['response']
            if record['response_hash'] != hashlib.sha256(json.dumps(response,
                sort_keys=True).encode()).hexdigest():
                raise ValueError('Replay response mismatch')
            return response
        token = os.environ.get(self.api_key_env)
        if not token:
            raise ValueError(f'Missing environment variable {self.api_key_env}')
        req = Request(self.endpoint,
            data=json.dumps(request).encode(),
            method='POST',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'})
        with urlopen(req, timeout=self.timeout) as stream:
            raw = json.load(stream)
        if raw['choices'][0].get('finish_reason') not in (None, 'stop'):
            raise ValueError('Model output is incomplete; check output token limit')
        result = json.loads(raw['choices'][0]['message']['content'])
        if not isinstance(result, dict):
            raise ValueError('Agent must return a JSON object')
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(request_hash=key, response=result, response_hash=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()), ensure_ascii=False),
                encoding='utf-8')
        return result
