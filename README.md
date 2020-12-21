0. 대회 규격, 기술교류 발표자료등은 docs 폴더 참고


1. docker 파일 다운로드  
  (훈련된 모델 weight 파일은 도커 파일에 들어 있음)
    https://drive.google.com/file/d/14aCQvGoXbFlSyP3Qybvs6UAF0PfIT4HK/view?usp=sharing



2. 다운로드 받은 tar.gz  도커 파일을 이미지로 만들기  
    ```bash  
        sudo docker load -i 4th_tr1_est_kts2.tar.gz  
    ```  
    
    
3. 이미지를 실행 시켜 도커 컨테이너 띄우기

    ```bash  
        sudo docker run --name  vfnet4   --shm-size=1G --gpus=all -v /data/aichallenge:/dataset/4th-track1 -d 4th_tr1_public:latest  sleep infinity
    ```  

4. 도커 밖에서 실행시키기
    ```bash  
        sudo docker exec -it vfnet4 python /aichallenge/predict.py /dataset/4th-track1
    ```  


5. 도커 안으로 들어가서 결과 json 살펴보기
    ```bash  
        sudo docker exec -it vfnet4 bash
        vi /aichallenge/t1_res_U0000000217.json
    ```    
  
