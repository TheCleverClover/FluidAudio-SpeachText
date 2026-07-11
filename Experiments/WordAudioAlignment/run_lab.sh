#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
EXPERIMENT_ROOT="$ROOT/Experiments/WordAudioAlignment"
if [ -d /Applications/Xcode-beta.app/Contents/Developer ]; then
    DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-beta.app/Contents/Developer}
fi
export DEVELOPER_DIR

cd "$EXPERIMENT_ROOT"
BIN_DIR=$(xcrun swift build --product WordAudioLab --show-bin-path)
APP="$EXPERIMENT_ROOT/.build/WordAudioLab.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp "$BIN_DIR/WordAudioLab" "$APP/Contents/MacOS/WordAudioLab"
cp "$EXPERIMENT_ROOT/AppBundle/Info.plist" "$APP/Contents/Info.plist"
codesign --force --sign - "$APP"
open "$APP"

echo "$APP"
