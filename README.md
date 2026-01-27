# CudaBridge

Apple Silicon Mac에서 외부 NVIDIA eGPU를 통해 CUDA 연산을 가능하게 하는 오픈소스 프로젝트

## 개요

CudaBridge는 Apple Silicon (M1/M2/M3/M4) 기반 Mac에서 USB4/Thunderbolt를 통해 연결된
외부 NVIDIA GPU에서 CUDA 연산을 수행할 수 있게 해주는 소프트웨어 스택입니다.

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Applications (CUDA Code)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                     CudaBridge Userspace Library                         │
│                    (libcudabridge.dylib / API Layer)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                        CUDA Runtime Bridge                               │
│              (cudaMemcpy, cudaMalloc, kernel launch 등)                  │
├─────────────────────────────────────────────────────────────────────────┤
│                      GPU Driver Compatibility Layer                      │
│                  (NVIDIA GPU 초기화, 명령 큐, 메모리 관리)                 │
├─────────────────────────────────────────────────────────────────────────┤
│                    PCIe Tunneling Over USB4/Thunderbolt                  │
│                  (PCIe 패킷 캡슐화, DMA 처리, 인터럽트)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                       macOS DriverKit Extension                          │
│                    (USB4/Thunderbolt 하드웨어 접근)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                         Hardware Layer                                   │
│            Apple Silicon ←→ USB4/TB4 ←→ eGPU Enclosure ←→ NVIDIA GPU    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 주요 컴포넌트

### 1. USB4/Thunderbolt PCIe 터널링 드라이버
- Apple Silicon의 USB4 컨트롤러와 직접 통신
- PCIe 트랜잭션을 USB4 프로토콜로 캡슐화
- DMA(Direct Memory Access) 처리
- 핫플러그 지원

### 2. NVIDIA GPU 호환 레이어
- GPU 초기화 및 리셋 시퀀스
- BAR(Base Address Register) 메모리 매핑
- 명령 제출 및 완료 처리
- 전력 관리

### 3. CUDA Runtime Bridge
- CUDA Runtime API 호환 인터페이스
- 커널 컴파일 및 실행
- 메모리 관리 (호스트 ↔ 디바이스)
- 스트림 및 이벤트 동기화

### 4. Userspace Library
- 애플리케이션 개발자용 API
- 기존 CUDA 코드와의 호환성
- 디버깅 및 프로파일링 도구

## 시스템 요구사항

### 하드웨어
- Apple Silicon Mac (M1, M2, M3, M4 시리즈)
- USB4 또는 Thunderbolt 4 포트
- 호환되는 eGPU 인클로저 (Thunderbolt 3/4 지원)
- NVIDIA GPU (Ampere, Ada Lovelace, 또는 최신 아키텍처 권장)

### 소프트웨어
- macOS 13.0 (Ventura) 이상
- Xcode Command Line Tools
- CMake 3.20+

## 빌드 방법

```bash
# 저장소 클론
git clone https://github.com/yourusername/CudaBridge.git
cd CudaBridge

# 빌드 디렉토리 생성
mkdir build && cd build

# CMake 구성
cmake ..

# 빌드
make -j$(sysctl -n hw.ncpu)

# 설치 (관리자 권한 필요)
sudo make install
```

## 사용 예시

```c
#include <cudabridge.h>

int main() {
    // CudaBridge 초기화
    cbInit();

    // eGPU 디바이스 열거
    int deviceCount;
    cbGetDeviceCount(&deviceCount);

    // 디바이스 선택
    cbSetDevice(0);

    // 메모리 할당
    float *d_data;
    cbMalloc((void**)&d_data, 1024 * sizeof(float));

    // 데이터 전송
    float h_data[1024];
    cbMemcpy(d_data, h_data, 1024 * sizeof(float), CB_MEMCPY_HOST_TO_DEVICE);

    // CUDA 커널 실행
    dim3 blocks(1);
    dim3 threads(1024);
    cbLaunchKernel("my_kernel", blocks, threads, ...);

    // 동기화
    cbDeviceSynchronize();

    // 정리
    cbFree(d_data);
    cbShutdown();

    return 0;
}
```

## 프로젝트 구조

```
CudaBridge/
├── src/
│   ├── kernel/           # 커널 레벨 코드
│   │   ├── usb4/         # USB4 컨트롤러 드라이버
│   │   ├── pcie/         # PCIe 터널링
│   │   └── gpu/          # GPU 추상화 레이어
│   ├── driver/           # DriverKit 확장
│   │   ├── thunderbolt/  # Thunderbolt 프로토콜
│   │   └── nvidia/       # NVIDIA GPU 드라이버
│   ├── cuda/             # CUDA 호환 레이어
│   │   ├── runtime/      # CUDA Runtime API
│   │   ├── compiler/     # PTX/SASS 컴파일러 인터페이스
│   │   └── memory/       # 메모리 관리
│   └── userspace/        # 유저스페이스 라이브러리
│       ├── lib/          # 라이브러리 구현
│       ├── include/      # 공개 헤더
│       └── examples/     # 예제 코드
├── docs/                 # 문서
├── tests/                # 테스트
└── scripts/              # 빌드/설치 스크립트
```

## 알려진 제한사항

1. **macOS SIP**: System Integrity Protection 비활성화 필요
2. **드라이버 서명**: 개발 중에는 개발자 서명 필요
3. **대역폭**: USB4의 이론적 최대 대역폭은 40Gbps (PCIe 3.0 x4 수준)
4. **호환성**: 모든 CUDA 기능이 지원되지 않을 수 있음

## 기여하기

기여를 환영합니다! Pull Request를 보내기 전에 이슈를 먼저 생성해 주세요.

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조

## 면책 조항

이 프로젝트는 실험적이며 교육 목적으로 제공됩니다. 프로덕션 환경에서의 사용은 권장하지 않습니다.
Apple, NVIDIA, CUDA는 각 회사의 상표입니다.
