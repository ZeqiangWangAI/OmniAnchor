"""Render the repository's single-line BibTeX fields into an auditable reference draft."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def authors(value):
    if value.startswith('{'):
        return value.strip('{}')
    names = []
    for name in value.split(' and '):
        surname, given = name.split(', ', 1)
        initials = []
        for token in given.split():
            initials.append('-'.join(part[0]+'.' for part in token.split('-') if part))
        names.append(surname+', '+' '.join(initials))
    return names[0] if len(names) == 1 else ', '.join(names[:-1])+', & '+names[-1]


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'paper/references.bib'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    for block in re.split(r'(?=^@)', source.read_text(), flags=re.M):
        if not block.strip():
            continue
        match = re.match(r'@(\w+)\{([^,]+),', block)
        fields = dict(re.findall(r'^\s*(\w+)\s*=\s*\{(.*)\},?\s*$', block, re.M))
        if not match or not {'author', 'title'} <= fields.keys():
            raise ValueError('Unsupported bibliography entry.')
        kind, key = match.groups()
        clean = lambda value: value.replace('{', '').replace('}', '').replace('--', '–')
        author = authors(fields['author']); year = fields.get('year', 'n.d.')
        title = clean(fields['title'])
        text = f'{author} ({year}). {title}.'
        if 'journal' in fields:
            text += f" *{clean(fields['journal'])}*"
            if 'volume' in fields:
                text += f", *{fields['volume']}*"
                if 'number' in fields: text += f"({fields['number']})"
            if 'articleno' in fields: text += ', Article '+fields['articleno']
            elif 'pages' in fields: text += ', '+clean(fields['pages'])
            text += '.'
        elif 'booktitle' in fields:
            text += ' In *'+clean(fields['booktitle'])+'*'
            if 'pages' in fields: text += ' (pp. '+clean(fields['pages'])+')'
            text += '.'
        elif 'eprint' in fields:
            text += ' arXiv.'
        elif 'publisher' in fields:
            text += ' '+clean(fields['publisher'])+'.'
        url = 'https://doi.org/'+fields['doi'] if 'doi' in fields else fields.get('url')
        if url: text += f' [Source]({url})'
        entries.append({'key': key, 'author': author, 'year': year, 'title': title, 'reference': text})
    entries.sort(key=lambda row: (row['author'].casefold(), row['year'], re.sub(r'\W', '', row['title']).casefold()))
    for entry in entries:
        peers = [r for r in entries if (r['author'], r['year']) == (entry['author'], entry['year'])]
        if len(peers) > 1:
            suffix = ('-' if entry['year'] == 'n.d.' else '')+chr(ord('a')+peers.index(entry))
            entry['reference'] = entry['reference'].replace('('+entry['year']+').', '('+entry['year']+suffix+').', 1)
    (out/'references.md').write_text('## References\n\n'+'\n\n'.join(row['reference'] for row in entries)+'\n')
    (out/'manifest.json').write_text(json.dumps({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'entries': len(entries), 'keys': [r['key'] for r in entries],
        'scope': 'Formatting draft from previously verified BibTeX metadata; missing publication fields remain absent; final APA editorial review pending'}, indent=2))


if __name__ == '__main__':
    main()
