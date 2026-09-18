from pathlib import Path

from scripts.prepare_vatex_full import stage
from omnianchor.io import load_samples, read_json
from omnianchor.types import Part, Sample


def test_staging_resolves_identical_media_from_deeper_shards(tmp_path):
    media = tmp_path/'original'/'video.mp4'
    media.parent.mkdir()
    media.write_bytes(b'path identity fixture, not a video experiment')
    sample = Sample(id='v', parts=(Part(type='video', path=str(media)),),
                    language='en', group_id='source', metadata={'split':'test'})
    path = tmp_path/'prepared'/'test'/'shards'/'samples.json'
    stage([sample],path)
    assert load_samples(path)[0] == sample
    assert not Path(read_json(path)[0]['parts'][0]['path']).is_absolute()
