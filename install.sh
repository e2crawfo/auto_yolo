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
cd  ../
pip install -r requirements.txt
pip install -e .
cd auto_yolo/tf_ops/render_sprites/
make
cd ../../../
