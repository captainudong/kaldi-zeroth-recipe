# Kaldi-based ASR Zeroth Recipe 실습 기록 (2022.03.)  
AWS EC2 환경(Ubuntu 18.04 / g4dn.4xlarge / 600gb 스토리지)에서 학습을 진행했습니다.  
Kaldi로 학습을 진행하기 위해서는 60gb 이상의 고사양 메모리가 필요합니다.(N-gram에 따라 다름)

### kaldi 설치
kaldi를 clone하여 우선 설치합니다.
```bash
cd kaldi
~/kaldi$ cd tools
~/kaldi/tools$ nano INSTALL
~/kaldi/tools$ CXX=g++-4.8 extras/check_dependencies.sh
~/kaldi/tools$ sudo apt-get install g++ sox subversion
~/kaldi/tools$ CXX=g++-4.8 extras/check_dependencies.sh
~/kaldi/tools$ sudo apt-get install g++
~/kaldi/tools$ CXX=g++-4.8 extras/check_dependencies.sh
~/kaldi/tools$ extras/install_mkl.sh
~/kaldi/tools$ CXX=g++-4.8 extras/check_dependencies.sh
~/kaldi/tools$ nano INSTALL
~/kaldi/tools$ make -j 16 # 뒤 숫자는 cpu 개수
~/kaldi/tools$ extras/install_irstlm.sh
~/kaldi/tools$ cd ..

~/kaldi$ cd src
~/kaldi/src$ nano INSTALL
~/kaldi/src$ ./configure --shared
~/kaldi/src$ nano INSTALL
~/kaldi/src$ make depend -j 16
~/kaldi/src$ nano INSTALL
~/kaldi/src$ make -j 16
```

### 로케일 설정
```bash
locale
sudo apt-get install language-pack-ko
sudo locale-gen ko_KR.UTF-8
sudo dpkg-reconfigure locales # ko_KR.UTF-8을 스페이스로 선택 후 엔터
sudo update-locale LANG=ko_KR.UTF-8 LC_MESSAGES=POSIX
exit #ssh 재접속
locale #설정이 제대로 되었는지 확인
```

### 라이브러리 설치
```bash
sudo apt-get update;
sudo apt-get install -y zlib1g-dev make automake autoconf libtool;
sudo apt-get install -y subversion;
sudo apt-get install -y libatlas3-base;
sudo apt-get install -y build-essential;
sudo apt-get install -y python;
sudo apt-get install -y python-pip python-dev;
sudo python -m pip install awscli;
sudo apt-get install -y unzip;
sudo apt-get install -y flac;
sudo apt-get install -y sox;
sudo apt-get install -y libsox-fmt-all;
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py";
sudo python get-pip.py;
sudo python3 get-pip.py;
sudo apt-get install -y gawk;

sudo apt-get install parallel;
sudo add-apt-repository ppa:openjdk-r/ppa;
sudo apt-get update;
sudo apt-get install g++ openjdk-7-jdk python-dev python3-dev;
sudo python3 -m pip install JPype1-py3;
sudo python3 -m pip install konlpy;
sudo apt-get install curl;
sudo -s;
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh);
```

### gcc 설치 및 심볼릭 링크 설정
```bash
#gcc, g++ 7.4 이하 버전
sudo apt install gcc-7 g++-7

#심볼릭 링크
sudo ln -s /usr/bin/gcc-7 /usr/local/cuda/bin/gcc
sudo ln -s /usr/bin/g++-7 /usr/local/cuda/bin/g++
```

### srilm 설치
http://www.speech.sri.com/projects/srilm/download.html 에 접속하여 다운로드 후 아래 코드 실행
```bash
cd kaldi/tools
./install_srilm.sh
```

### zeroth 다운로드 및 심볼릭 링크 설정
```bash
cd ~
git clone https://github.com/goodatlas/zeroth
cd zeroth/s5
rm steps utils
ln -s /home/ubuntu/kaldi/egs/wsj/s5/steps steps
ln -s /home/ubuntu/kaldi/egs/wsj/s5/utils utils
```

### zeroth/s5/path.sh 파일 수정
KALDI_ROOT 에 실제 경로를 지정합니다.
```bash
export KALDI_ROOT=/home/ubuntu/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file$
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=ko_KR.UTF-8
```

### run_tdnn_*.sh 수정
gpu 1개를 사용했기 때문에 num_jobs 변수를 수정했습니다.
```python
# training options
num_jobs_initial=1
num_jobs_final=1
num_epochs=4
minibatch_size=128
initial_effective_lrate=0.0015
final_effective_lrate=0.0002
remove_egs=true
```

### run_openslr.sh 수정
한번에 학습이 이어지도록 하단 GMM과 DNN 트레이닝 사이 exit 명령어를 주석처리했습니다.

### 실행
```bash
nvidia-smi --compute-mode=3
cd /zeroth/s5

nohup ./run_openslr.sh & #백그라운드에서 실행하며 로그는 nohup.out에 기록
disown #ssh 연결이 끊어져도 프로세스가 살아있음

tail -f ./nohup.out #로그 실시간 보기
```

### export.sh 수정
학습이 완료된 모델을 따로 저장하기 위해 경로를 수정합니다.
```bash
final_graph_dir=/home/ubuntu/_prjs_/zeroth/s5/exp/chain_rvb/tree_a/graph_tgsmall
final_model_dir=/home/ubuntu/_prjs_/zeroth/s5/exp/chain_rvb/tdnn1n_rvb_online
small_lm=/home/ubuntu/_prjs_/zeroth/s5/data/lang_test_tgsmall/G.fst
large_lm=/home/ubuntu/_prjs_/zeroth/s5/data/lang_test_fglarge/G.carpa
```

### 참고
메모리가 부족하다면 max_shuffle_jobs_run 변수를 수정하는 것이 도움이 됩니다.
```python
#egs/wsj/s5/steps/nnet3/chain/get_egs.sh
max_shuffle_jobs_run=10 #50->10 으로 줄이기
```

디코딩 그래프를 정적 그래프(HCLg.fst)가 아닌 동적 그래프(HCLr.fst, Gr.fst)를 사용하면 크기가 작아집니다.  
mkgraph_lookahead.sh를 통해 동적 그래프를 생성할 수 있습니다.
