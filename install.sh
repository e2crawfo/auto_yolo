# Check tensorflow version.

MAX_VERSION=1.13
VALID_TF_VERSION=$(python -c "import tensorflow as tf; from pkg_resources import parse_version; print(parse_version(tf.VERSION) < parse_version('$MAX_VERSION'))")

if [ $VALID_TF_VERSION != "True" ]; then
  TF_VERSION=$(python -c "import tensorflow as tf; print(tf.VERSION)")
  echo "Version $TF_VERSION too new, must be < $MAX_VERSION"
fi

GIT_BRANCH=$(git branch | sed -n '/\* /s///p')
echo "*** git branch for auto_yolo is <"$GIT_BRANCH">."

# Install dps
echo "*** Installing dps on branch "$GIT_BRANCH
git clone https://github.com/e2crawfo/dps.git
cd dps
git checkout "$GIT_BRANCH"
echo "*** git branch for dps is <"$GIT_BRANCH">."
pip install -r requirements.txt
pip install -e .

# Install auto_yolo
cd  ..
pip install -r requirements.txt
pip install -e .
cd auto_yolo/tf_ops/render_sprites/
make
cd ../resampler_edge
make
cd ../../..
