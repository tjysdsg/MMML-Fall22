#!/usr/bin/bash

# call this script in project root directory

# sudo apt install p7zip-full

out_dir=webqa_data
mkdir -p $out_dir
cd $out_dir

end=10
x=1
until [ $x -eq $end ]; do
  if [ ! -f imgs.7z.00$x ]; then
    wget http://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/WebQA_imgs_7z_chunks/imgs.7z.00$x || exit 1
  fi
  x=$(($x + 1))
done

end=52
until [ $x -eq $end ]; do
  if [ ! -f imgs.7z.0$x ]; then
    wget http://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/WebQA_imgs_7z_chunks/imgs.7z.0$x || exit 1
  fi
  x=$(($x + 1))
done

if [ ! -f imgs.lineidx ]; then
  wget http://tiger.lti.cs.cmu.edu/yingshac/WebQA_data_first_release/imgs.lineidx || exit 1
fi

# unzip images
if [ ! -f imgs.tsv ]; then
  7z x imgs.7z.001 || exit 1
fi

# decode images
python ../utils/decode_base64_images.py --input=imgs.tsv --output=images