![](docs/img/logo_hippoLBM.png)

# hippoLBM

## Installation guidelines

### Add Onika

```
git clone https://github.com/Collab4exaNBody/onika.git
cd onika
mkdir build
cd build
cmake ..
make install -j 10
cd ../install
export onika_DIR=${PWD}
spack install yaml-cpp@0.6.3
spack load yaml-cpp@0.6.3
```

## pour moi

export ONIKA_CONFIG_PATH=/home/rp269144/codes/hipoLBM/data/config
