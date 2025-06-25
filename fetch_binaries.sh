#!/bin/env sh

set -e

GDRIVE_FID=1hwNZbAEEJPoFkvYoq-J4yadFsbsnbHdU
OUTDIR=$(pwd)/artefacts
VENV_DIR=$(mktemp -d)
PIP=$VENV_DIR/bin/pip

if [ "$1" = "--out-dir" ] && [ -n "$2" ]; then
    OUTDIR="$2"
    shift 2
fi


ARTEFACT=$OUTDIR/artefacts-bin.tar.xz

python3 -m venv "$VENV_DIR"

$PIP install --upgrade pip
$PIP install gdown

mkdir -p $OUTDIR
echo "Downloading from Google Drive..."
$VENV_DIR/bin/gdown $GDRIVE_FID -O $ARTEFACT
mkdir -p $OUTDIR
tar -xvf $ARTEFACT -C $OUTDIR

rm -rf "$VENV_DIR"
echo "Done"

