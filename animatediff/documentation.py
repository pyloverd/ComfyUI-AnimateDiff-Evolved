from typing import Union

def short_desc(desc):
    return str(desc)

def coll(text: str):
    return text

descriptions = {}

def as_markdown(entry, depth=0):
    if isinstance(entry, dict):
        sections = []
        for name, value in entry.items():
            if name == "collapsed":
                continue
            heading = "#" * min(depth + 2, 6)
            rendered = as_markdown(value, depth=depth + 1)
            sections.append(f"{heading} {name}\n\n{rendered}" if rendered else f"{heading} {name}")
        return "\n\n".join(sections)
    if isinstance(entry, list):
        return "\n\n".join(as_markdown(item, depth=depth) for item in entry)
    return str(entry)


def register_description(node_id: str, desc: Union[list, dict]):
    descriptions[node_id] = desc


def format_descriptions(nodes):
    for node_id, description in descriptions.items():
        nodes[node_id].DESCRIPTION = as_markdown(description)


class DocHelper:
    def __init__(self):
        self.actual_dict = {}
    
    def add(self, add_dict):
        self.actual_dict.update(add_dict)
        return self

    def get(self):
        return self.actual_dict
    
    @staticmethod
    def combine(*args):
        docs = DocHelper()
        for doc in args:
            docs.add(doc)
        return docs.get()
