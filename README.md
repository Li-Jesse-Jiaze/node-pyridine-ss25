# node-pyridine-ss25

See `demo.ipynb`

## Setup Environment

Use `uv`:

```bash
# Install uv first, if you haven't
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

<details>
<summary>(Optional) Add `HSL`</summary>

```shell
git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
```

Obtain a tarball with Coin-HSL source code from https://licences.stfc.ac.uk/product/coin-hsl

Inside the ThirdParty-HSL directory, unpack this tarball via

```shell
gunzip coinhsl-x.y.z.tar.gz
tar xf coinhsl-x.y.z.tar
```

Rename the directory `coinhsl-x.y.z` to `coinhsl`, or set a symbolic link:

```shell
ln -s coinhsl-x.y.z coinhsl
```

Run `./configure`. Use `./configure --help` to see available options.

```shell
./configure
```

Build and install

```shell
make
make install
```

Normally in this path

```shell
cd /usr/local/lib
ln -s libcoinhsl.so libhsl.so
```

Add this line to `.venv/bin/activate`

```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

</details>
