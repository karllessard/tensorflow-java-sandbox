#!/bin/bash
# Script to build native TensorFlow libraries
set -eu

KERNEL=(`uname -s | tr [A-Z] [a-z]`)
ARCH=(`uname -m | tr [A-Z] [a-z]`)
case $KERNEL in
    darwin)
        OS=macosx
        ;;
    mingw32*)
        OS=windows
        KERNEL=windows
        ARCH=x86
        ;;
    mingw64*)
        OS=windows
        KERNEL=windows
        ARCH=x86_64
        ;;
    *)
        OS=$KERNEL
        ;;
esac
case $ARCH in
    arm*)
        ARCH=arm
        ;;
    aarch64*)
        ARCH=arm64
        ;;
    i386|i486|i586|i686)
        ARCH=x86
        ;;
    amd64|x86-64)
        ARCH=x86_64
        ;;
esac
PLATFORM=$OS-$ARCH
EXTENSION=
echo "Detected platform \"$PLATFORM\""

while [[ $# > 0 ]]; do
    case "$1" in
        -platform=*)
            PLATFORM="${1#-platform=}"
            ;;
        -platform)
            shift
            PLATFORM="$1"
            ;;
        -extension=*)
            EXTENSION="${1#-extension=}"
            ;;
        -extension)
            shift
            EXTENSION="$1"
            ;;
        clean)
            echo "Cleaning build"
            rm -Rf build
            ;;
    esac
    shift
done

echo -n "Building for platform \"$PLATFORM\""
if [[ -n "$EXTENSION" ]]; then
    echo -n " with extension \"$EXTENSION\""
fi
echo

export PYTHON_BIN_PATH=$(which python)
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_ENABLE_XLA=0
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_DOWNLOAD_CLANG=0
export TF_NEED_MPI=0
export CC_OPT_FLAGS=-O3
export TF_SET_ANDROID_WORKSPACE=0

TENSORFLOW_VERSION=2.0.0-rc0

mkdir -p "build/$PLATFORM$EXTENSION"
cd "build/$PLATFORM$EXTENSION"

if [[ ! -e "tensorflow-$TENSORFLOW_VERSION.tar.gz" ]]; then
    curl -L -o "tensorflow-$TENSORFLOW_VERSION.tar.gz" "https://github.com/tensorflow/tensorflow/archive/v$TENSORFLOW_VERSION.tar.gz"
fi
echo "48ddba718da76df56fd4c48b4bbf4f97f254ba269ec4be67f783684c75563ef8 tensorflow-$TENSORFLOW_VERSION.tar.gz" | sha256sum -c -

echo "Decompressing archives"
tar --totals -xzf tensorflow-$TENSORFLOW_VERSION.tar.gz

# Assume Bazel is available in the path: https://www.tensorflow.org/install/source
cd tensorflow-$TENSORFLOW_VERSION

# Run the build for both the C and Java APIs
bash configure
bazel build --config opt //tensorflow:tensorflow //tensorflow/java:tensorflow

# Normalize some paths with symbolic links
ln -sf tensorflow-$TENSORFLOW_VERSION ../tensorflow
ln -sf libtensorflow.so.${TENSORFLOW_VERSION%-*} bazel-bin/tensorflow/libtensorflow.so
ln -sf libtensorflow.so.${TENSORFLOW_VERSION%-*} bazel-bin/tensorflow/libtensorflow.so.2

# Gather Java source files from everywhere
mkdir -p ../java
cp -r tensorflow/java/src/gen/java/* ../java
cp -r tensorflow/java/src/main/java/* ../java
# cp -r tensorflow/contrib/android/java/* ../java
# cp -r tensorflow/lite/java/src/main/java/* ../java
cp -r bazel-genfiles/tensorflow/java/ops/src/main/java/* ../java
cp -r bazel-genfiles/tensorflow/java/_javac/tensorflow/libtensorflow_sourcegenfiles/* ../java

# Work around loader bug in NativeLibrary.java
sed -i="" '/TensorFlow.version/d' ../java/org/tensorflow/NativeLibrary.java

cd ../..
