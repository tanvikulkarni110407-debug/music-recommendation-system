import os
import json
from collections import defaultdict


class LocalCollection:
    def __init__(self, name, base_dir="data"):
        self.name = name
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, f"{self.name}.json")

    def _load(self):
        if not os.path.exists(self.path):
            return []

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, data):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def insert_one(self, document):
        data = self._load()
        data.append(document)
        self._save(data)

        class Result:
            inserted_id = len(data) - 1

        return Result()

    def insert_many(self, documents):
        data = self._load()
        data.extend(documents)
        self._save(data)

    def find(self, query=None):
        data = self._load()
        query = query or {}

        results = []
        for item in data:
            if all(item.get(k) == v for k, v in query.items()):
                results.append(item)

        return results

    def find_one(self, query=None):
        results = self.find(query)
        return results[0] if results else None

    def count_documents(self, query=None):
        return len(self.find(query))

    def delete_many(self, query=None):
        data = self._load()
        query = query or {}

        remaining = [
            item for item in data
            if not all(item.get(k) == v for k, v in query.items())
        ]

        self._save(remaining)


class LocalDB:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self._collections = {}

    def __getitem__(self, name):
        if name not in self._collections:
            self._collections[name] = LocalCollection(
                name,
                self.base_dir
            )
        return self._collections[name]