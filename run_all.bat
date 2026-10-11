@echo off
REM Runs every experiment in order. Each run resumes if it was interrupted (--resume).
REM Edit EPOCHS if the time estimate printed at the start of run 1 is too long for tonight.
set EPOCHS=50

REM ---- Tonight: 4 baselines
python train.py --model resnet50        --mode baseline   --epochs %EPOCHS% --resume || goto :err
python train.py --model efficientnet_b0 --mode baseline   --epochs %EPOCHS% --resume || goto :err
python train.py --model mobilenetv3     --mode baseline   --epochs %EPOCHS% --resume || goto :err
python train.py --model vit_small       --mode baseline   --epochs %EPOCHS% --resume || goto :err

REM ---- Overnight: proposed BG-Suppress
python train.py --model resnet50        --mode bgsuppress --epochs %EPOCHS% --resume || goto :err
python train.py --model efficientnet_b0 --mode bgsuppress --epochs %EPOCHS% --resume || goto :err
python train.py --model mobilenetv3     --mode bgsuppress --epochs %EPOCHS% --resume || goto :err
python train.py --model vit_small       --mode bgsuppress --epochs %EPOCHS% --resume || goto :err

REM ---- Tomorrow: ablation on ResNet-50
python train.py --model resnet50        --mode bgrand     --epochs %EPOCHS% --resume || goto :err
python train.py --model resnet50        --mode attloss    --epochs %EPOCHS% --resume || goto :err

python evaluate.py --all || goto :err
python make_figures.py || goto :err
echo ALL DONE. See results\results.csv and results\*.png
goto :eof

:err
echo A step failed. Fix the error above, then run this file again - finished runs are skipped quickly via --resume.
exit /b 1
