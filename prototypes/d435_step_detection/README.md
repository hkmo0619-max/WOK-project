\# Intel RealSense D435 Step Detection Prototype



WoKi 프로젝트에서 Intel RealSense D435를 이용해 단차의 경계와 높이를 검출하기 위해 개발한 standalone Python prototype입니다.



ROS2에 통합하기 전 Windows 환경에서 D435의 RGB 및 Depth 데이터를 이용하여 단차 검출·높이 측정 알고리즘을 개발하고 검증한 코드를 정리했습니다.



\*\*참고: 본 프로토타입의 개발 이력은 보존된 개발 버전을 기반으로 정리했습니다.\*\*



\## 1. Directory Structure



```text

d435\_step\_detection/

├── README.md

├── requirements.txt

├── d435\_boundary\_detector.py

├── d435\_step\_height.py

└── measurement\_tools/

&#x20;   ├── automatic\_step\_height.py

&#x20;   ├── manual\_auto\_step\_height.py

&#x20;   └── step\_height\_core.py

