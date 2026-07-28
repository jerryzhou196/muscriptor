#!/usr/bin/env bash
#
# Regenerates web/public/og-image.png — the social-preview card unfurled by
# Slack, X, LinkedIn etc. (see the og: tags in web/index.html).
#
# Rerun this whenever the piano roll's look changes: replace screenshot.png
# with a fresh grab of a transcription and run ./make.sh.
#
# Needs: imagemagick (brew install imagemagick) and uv, for the one-off
# fontTools call below.

set -euo pipefail
cd "$(dirname "$0")"

OUT=../public/og-image.png

# 1200x630 (1.91:1) is the Open Graph standard. X crops it to 2:1, so keep the
# text well inside the vertical middle and away from the left/right edges.
WIDTH=1200
HEIGHT=630

# The self-hosted Satoshi is a variable woff2, which ImageMagick can't read.
# Pin it to the two weights the card uses and write plain TTFs to a temp dir.
FONTS=$(mktemp -d)
trap 'rm -rf "$FONTS"' EXIT

uv run --quiet --with fonttools --with brotli python - "$FONTS" <<'PY'
import sys
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

out = sys.argv[1]
for weight in (700, 500):
    font = instancer.instantiateVariableFont(
        TTFont("../public/fonts/Satoshi-Variable.woff2"), {"wght": weight}
    )
    font.flavor = None  # plain TTF, not woff2
    font.save(f"{out}/Satoshi-{weight}.ttf")
PY

# The screenshot is 16:9-ish, so fill-and-centre-crop rather than letterbox.
# The 20% darken is what keeps the white type legible over the yellow notes.
magick screenshot.png \
  -resize "${WIDTH}x${HEIGHT}^" -gravity center -extent "${WIDTH}x${HEIGHT}" \
  -fill '#111216' -colorize 20% \
  -gravity center \
  -font "$FONTS/Satoshi-700.ttf" -pointsize 150 -kerning -2 \
  -fill '#edecf0' -annotate +0-45 'MuScriptor' \
  -font "$FONTS/Satoshi-500.ttf" -pointsize 54 -kerning 1 \
  -fill '#edecf0' -annotate +0+75 'Audio to MIDI converter' \
  "$OUT"

echo "wrote $OUT"
