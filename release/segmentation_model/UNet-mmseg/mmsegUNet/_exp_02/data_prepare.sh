echo hi

cd /tf/P_stage/P_stage_segmentation/segmentation/input/data && wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tBpoD5Z7zh6sOBh-v45GwhH1ZcJgXIFN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tBpoD5Z7zh6sOBh-v45GwhH1ZcJgXIFN" -O ./mmseg.tar && rm -rf ~/cookies.txt && tar -xvf mmseg.tar
