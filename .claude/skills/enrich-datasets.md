# Enrich Test Datasets

Use this skill when you need to add more test data — prompts, reference images, or video clips — to the load testing harness. It uses the storyboard MCP to generate real visual assets and prevents duplicates by checking the manifest.

## When to Use

- "add more prompts" / "we need more test data"
- "generate test images for i2v"
- "create video clips for v2v testing"
- "enrich the dataset" / "expand test coverage"
- Before a major test run when current datasets feel thin

## Prerequisites

- Storyboard MCP must be connected (`mcp__storyboard__*` tools available)
- `DAYDREAM_API_KEY` set (for storyboard to generate assets)

## Process

### 1. Read the manifest

```bash
cat config/datasets/manifest.yaml
```

Note what already exists — pool names, image IDs, clip IDs, tags, counts.

### 2. Identify gaps

Check what's missing based on the request:
- **Prompts:** Are all categories covered? (nature, urban, abstract, people, animals, weather, sci-fi, fantasy, etc.) Are there enough per pool (target: 20-30)?
- **Images:** Are there reference images for each prompt pool? (need at least 5 per pool for i2v)
- **Clips:** Are there video clips with varied motion types? (static, slow pan, fast action, scene change)

### 3. Generate new prompts (if needed)

Write new prompts that are:
- **Distinct** from existing ones (check the .yaml files, not just the manifest)
- **Varied** in complexity (simple scenes, complex multi-element scenes)
- **Specific** enough to produce recognizable output (not vague)
- **Grouped** by theme into a new or existing pool

Add to existing pool file or create a new one:

```yaml
# config/prompts/{pool_name}.yaml
prompts:
  - "new prompt here"
  - "another new prompt"
```

### 4. Generate reference images (if needed)

Use the storyboard MCP to create test images:

```
mcp__storyboard__create_media({
  prompt: "the exact prompt text",
  type: "image",
  model: "flux-dev",
  width: 512,
  height: 512
})
```

Save the resulting image URL/file to `config/datasets/images/` with a descriptive name.

### 5. Generate video clips (if needed)

For short test clips, use storyboard MCP:

```
mcp__storyboard__create_media({
  prompt: "the exact prompt text",
  type: "video",
  model: "wan",
  width: 512,
  height: 512
})
```

Or for scope-specific clips, use the SDK stream approach:
1. Start a t2v stream via SDK with the prompt
2. Capture output frames for 10s
3. Save as MP4

### 6. Update the manifest

After adding any assets, update `config/datasets/manifest.yaml`:
- Add new pool entries under `prompts.pools`
- Add new image entries under `images.items`
- Add new clip entries under `clips.items`
- Include: id, file path, prompt used, tags, source, resolution

### 7. Update default.yaml if new pools were created

If you created a new prompt pool, add it to the `prompts_pools` list in the relevant scenario entries in `config/default.yaml`.

### 8. Run dedup check

```bash
python -c "
import yaml
from pathlib import Path

# Check for duplicate prompts across all pools
seen = {}
for f in Path('config/prompts').glob('*.yaml'):
    with open(f) as fh:
        pool = yaml.safe_load(fh).get('prompts', [])
    for p in pool:
        key = p.strip().lower()
        if key in seen:
            print(f'DUPLICATE: \"{p[:60]}\" in {f.name} and {seen[key]}')
        seen[key] = f.name
print(f'Total: {len(seen)} unique prompts across {len(list(Path(\"config/prompts\").glob(\"*.yaml\")))} pools')
"
```

### 9. Commit

```bash
git add config/prompts/ config/datasets/ config/default.yaml
git commit -s -m "dataset: add [description of what was added]"
```

## Guidelines

- **No duplicates:** Always check existing pools before adding. Similar is OK, identical is not.
- **Tag everything:** Tags enable filtered selection in future (e.g., "only water scenes for this run").
- **Source tracking:** Always record whether an asset came from `manual`, `storyboard_mcp`, `sdk_capture`, or `url_download`.
- **Resolution matters:** All images/clips should be 512x512 unless testing resolution-specific behavior.
- **Keep pools focused:** 20-30 prompts per pool. Split into a new pool rather than growing one past 30.
- **Stress pool is special:** Only edge cases (empty, very long, special chars, unicode, adversarial). Don't mix normal prompts in.
