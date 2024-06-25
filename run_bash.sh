sed -i '6c "init_cls" : 20,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
sed -i '7c "increment" : 20,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
nohup python main.py --config=exps/FeCAM_cifar100.json >> cifar100_0_20.log 2>&1 &
wait
echo "cifar100 0/20 done"
sed -i '6c "init_cls" : 10,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
sed -i '7c "increment" : 10,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
nohup python main.py --config=exps/FeCAM_cifar100.json >> cifar100_0_10.log 2>&1 &
wait
echo "cifar100 0/10 done"
sed -i '6c "init_cls" : 5,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
sed -i '7c "increment" : 5,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
nohup python main.py --config=exps/FeCAM_cifar100.json >> cifar100_0_5.log 2>&1 &
wait
echo "cifar100 0/5 done"
# sed -i '6c "init_cls" : 40,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
# sed -i '7c "increment" : 3,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
# nohup python main.py --config=exps/FeCAM_cifar100.json >> cifar100_40_3.log 2>&1 &
# wait
# echo "cifar100 40/3 done"
# sed -i '6c "init_cls" : 50,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
# sed -i '7c "increment" : 10,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
# nohup python main.py --config=exps/FeCAM_cifar100.json >> cifar100_50_10.log 2>&1 &
# wait
# echo "cifar100 50/10 done"
# sed -i '6c "init_cls" : 50,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
# sed -i '7c "increment" : 5,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_cifar100.json
# nohup python main.py --config=exps/FeCAM_cifar100.json >> cifar100_50_5.log     2>&1 &
# wait
# echo "cifar100 50/5 done"
# echo "cifar100 done"


sed -i '6c "init_cls" : 20,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
sed -i '7c "increment" : 20,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
nohup python main.py --config=exps/FeCAM_imagnet100.json >> imagnet100_0_20.log 2>&1 &
wait
echo "imagnet100 0/20 done"
sed -i '6c "init_cls" : 10,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
sed -i '7c "increment" : 10,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
nohup python main.py --config=exps/FeCAM_imagnet100.json >> imagnet100_0_10.log 2>&1 &
wait
echo "imagnet100 0/10 done"
sed -i '6c "init_cls" : 5,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
sed -i '7c "increment" : 5,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
nohup python main.py --config=exps/FeCAM_imagnet100.json >> imagnet100_0_5.log 2>&1 &
wait
echo "imagnet100 0/5 done"
# sed -i '6c "init_cls" : 40,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# sed -i '7c "increment" : 3,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# nohup python main.py --config=exps/FeCAM_imagnet100.json >> imagnet100_40_3.log 2>&1 &
# wait
# echo "imagnet100 40/3 done"
# sed -i '6c "init_cls" : 50,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# sed -i '7c "increment" : 10,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# nohup python main.py --config=exps/FeCAM_imagnet100.json >> imagnet100_50_10.log 2>&1 &
# wait
# echo "imagnet100 50/10 done"
# sed -i '6c "init_cls" : 50,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# sed -i '7c "increment" : 5,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# nohup python main.py --config=exps/FeCAM_imagnet100.json >> imagnet100_50_5.log 2>&1 &
# wait
# echo "imagnet100 50/5 done"
# echo "imagnet100 done"

# sed -i '6c "init_cls" : 100,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# sed -i '7c "increment" : 20,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# nohup python main.py --config=exps/FeCAM_imagnet100.json >> imagnet100_100_20.log 2>&1 &
# wait
# echo "imagnet100 100/20 done"
# sed -i '6c "init_cls" : 100,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# sed -i '7c "increment" : 10,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# nohup python main.py --config=exps/FeCAM_imagnet100.json >> imagnet100_100_10.log 2>&1 &
# wait
# echo "imagnet100 100/10 done"
# sed -i '6c "init_cls" : 100,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# sed -i '7c "increment" : 5,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_imagnet100.json
# nohup python main.py --config=exps/FeCAM_imagnet100.json >> imagnet100_100_5.log 2>&1 &
# wait
# echo "imagnet100 100/5 done"
# echo "tinyimagenet done"




# sed -i '28c "source_index" : 0,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# sed -i '29c "target_index" : 2' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_022.log 2>&1 &
# wait
# echo "office_31_022 done"

# sed -i '28c "source_index" : 1,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# sed -i '29c "target_index" : 0' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_120.log 2>&1 &
# wait
# echo "office_31_120 done"

# sed -i '28c "source_index" : 1,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# sed -i '29c "target_index" : 2' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_122.log 2>&1 &
# wait
# echo "office_31_122 done"

# sed -i '28c "source_index" : 2,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# sed -i '29c "target_index" : 0' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_220.log 2>&1 &
# wait
# echo "office_31_220 done"

# sed -i '28c "source_index" : 2,' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# sed -i '29c "target_index" : 1' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_221.log 2>&1 &
# wait
# echo "office_31_221 done"



# echo "run the ProCA-main"
# cd "/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main"
# source ~/anaconda3/bin/activate CIUDA
# sh ./bash/resnet_incre_copy.sh
# wait
# echo "ProCA-main done"

# cd /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main
# source ~/anaconda3/bin/activate tllib
# sed -i 's/"target_index" : 0/"target_index" : 1/g' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_221.log 2>&1 &
# wait
# echo "office_31_221 done"
# sed -i 's/"target_index" : 1/"target_index" : 3/g' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_223.log 2>&1 &
# wait
# echo "office_31_223 done"
# sed -i 's/"source_index" : 2,/"source_index" : 3,/g' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# sed -i 's/"target_index" : 3/"target_index" : 0/g' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_320.log 2>&1 &
# wait
# echo "office_31_330 done"
# sed -i 's/"target_index" : 0/"target_index" : 1/g' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_321.log 2>&1 &
# wait
# echo "office_31_321 done"
# sed -i 's/"target_index" : 1/"target_index" : 2/g' /public/home/imgbreaker/Desktop/FeCAM/FeCAM-main/exps/FeCAM_office_31.json
# nohup python main.py --config=exps/FeCAM_office_31.json > office_31_322.log 2>&1 &
# wait
# echo "office_31_322 done"

