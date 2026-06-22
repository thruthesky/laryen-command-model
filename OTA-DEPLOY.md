# LCM OTA(원격 업데이트) 배포 설계 — manifest 기반

LCM(`lcm.int8.onnx`)을 **Flutter 빌드에 임베드하지 않고**, 앱 실행 후 manifest 를 확인해
변경이 있으면 다운로드하는 구조의 SSOT. 두 팀 분석(A=본 세션 / B=다른 팀)을 교차 검토해
통합한 결과다. 실제 lib/ 통합·onnxruntime 도입·서버 배포는 **사용자 승인 차단지점**(§7).

관련 문서: [INTEGRATION.md](INTEGRATION.md)(3계층 classify·BPE), [README.md](README.md)(SML 설계).

---

## 0. 두 분석 교차 검토 — 합의 / 차이 / 통합

**두 분석은 큰 줄기에서 완전히 일치한다**(manifest 기반 · bundled fallback + 원격 최신 2단
구조 · version 비교 stale-while-revalidate · sha256 · 3자산 세트 · 빌드 디커플링 · 재훈련
필요/불필요 구분 · 호환 게이트 · CF 안전망). 서로의 강점을 흡수해 통합한다.

| 축 | 팀 A(본 세션) | 팀 B(다른 팀) | 통합 채택 |
|---|---|---|---|
| 무결성 | `sha256` 명시 | `sha256` + 자산별 해시 | **자산별 sha256**(B) |
| 호환 게이트 | `minClientCompat`(라리엔 `kClientCompat` 재사용) | `schema_version` + `min_app_version` | **둘 통합** → `schema_version`(decoder 호환) + `min_app_version`(앱 게이트) (§2) |
| SSOT 동기 | CI 가드 `sync_ssot.py --check` | `ssot_hash` manifest 동봉 | **둘 다**(빌드타임 CI + 런타임 hash 표기) |
| fallback | bundled 1개 | bundled + **직전 정상(N-1) 롤백** | **bundled + N-1 캐시 롤백**(B 강화) |
| 강제 무효화 | (없음) | `min_lcm_version`(구 모델 sunset) | **`min_lcm_version` 채택**(B) — 보안/치명 버그 모델 차단 |
| 릴리스 채널 | (없음) | `channel: stable` | **채택**(B) — 라리엔 staging/production 분리와 정합 |
| 자산 원자성 | head_specs 출력 매핑 어긋남 = silent wrong dispatch 경고 | labels_url 분리 배포 | **labels.json 원자 세트 + 경고**(A 근거 + B 형태) |
| 선례 | fast_path version·sherpa 다운로드 코드 인용 | (일반론) | **라리엔 기존 코드 재사용**(A) (§5) |

차이는 **상충이 아니라 상호 보완**이다. 통합본이 두 분석의 상위집합이다.

---

## 1. 핵심 원칙

1. **빌드 디커플링** — Flutter 빌드 ≠ LCM 빌드. LCM 은 런타임 다운로드 자산이다(앱
   재배포·스토어 심사 없이 갱신). 라리엔은 이미 sherpa STT 모델·fast_path.json 을 이 방식으로
   받는다(§5).
2. **2단 구조** — `bundled LCM`(오프라인/첫 실행/다운로드 실패 안전망) + `원격 최신 LCM`.
3. **3자산은 원자 세트** — `model(onnx)` + `tokenizer(vocab/merges)` + `labels(head_specs)` 는
   *반드시 같은 버전*. labels 인덱스가 어긋나면 argmax 결과를 *조용히 틀린 location* 으로
   디스패치한다(예: 인덱스 5 가 "강남"인지 "관악"인지 어긋남). sherpa 가 model+tokens 를 한
   번에 받는 것과 동일한 이유.
4. **graceful degradation** — 다운로드 실패·구버전·호환 불가여도 서비스는 안 끊긴다:
   `bundled → (LCM 우회) → 3차 CF(Gemini) 폴백`. "강제"는 하드 차단이 아니라 베스트-에포트
   최적화(받으면 0ms 온디바이스, 못 받으면 느린 CF).

---

## 2. manifest 스키마 (통합본)

`https://laryen.com/models/lcm/manifest.json` (또는 `laryen-manifest.yaml`).

```yaml
lcm:
  channel: stable            # stable | beta (라리엔 production/staging 정합)
  version: "1.2.3"           # 모델 가중치 버전 — stale-while-revalidate 비교 키
  schema_version: 4          # ★ decoder 호환: 앱이 이해하는 action/slot/head 구조 버전.
                             #   앱 bundled schema_version 보다 크면 = 새 action/slot 생김
                             #   → 앱 decoder 가 모름 → 다운로드 SKIP(앱 업데이트 필요).
  min_app_version: "1.3.9"   # 이 모델을 안전히 해석 가능한 최소 앱 버전(라리엔 pubspec version)
  min_lcm_version: "1.1.0"   # ★ 서버 강제 무효화 — 로컬이 이보다 낮으면 SML 비활성 + CF only
                             #   (치명 버그/보안 sunset 모델 차단)
  ssot_hash: "ab12…"         # 이 모델이 학습된 ssot.json 해시(클라 SSOT 동기 표기)
  files:
    model:     { url: ".../lcm-v1.2.3/lcm.int8.onnx", sha256: "…", bytes: 663000 }
    tokenizer: { url: ".../lcm-v1.2.3/tokenizer.json", sha256: "…" }  # vocab+merges 묶음
    labels:    { url: ".../lcm-v1.2.3/labels.json",    sha256: "…" }  # labels+head_specs+threshold+pad_len
```

> `labels.json` = 현재 [artifacts/golden_tokenize.json](artifacts/golden_tokenize.json) 의
> `labels` / `head_specs` / `threshold` / `pad_len` 키를 떼어낸 것. **이 4개가 모델과 한 세트.**

### 두 호환 축(혼동 금지)

| 축 | 무엇 | 어긋나면 | 조치 |
|---|---|---|---|
| `version` | 가중치 신선도 | 구 모델 사용 | 더 높으면 교체(stale-while-revalidate) |
| `schema_version` / `min_app_version` | **decoder 가 라벨/헤드를 해석 가능한가** | 새 action/slot 을 앱이 모름 → 디스패치 실패 | **다운로드 SKIP + 앱 업데이트 유도** |
| `min_lcm_version` | 서버측 강제 하한 | 로컬 모델이 너무 낡음 | **SML 비활성 → CF only** |

---

## 3. 앱 실행 시 런타임 흐름 (stale-while-revalidate)

```text
1. bundled LCM 로드(즉시 동작 — 부팅 막지 않음)
   └ 캐시에 직전 다운로드 모델 있으면 그것 우선(version 높은 쪽)
2. GET manifest.json (백그라운드)
3. 게이트 판정:
   a. manifest.schema_version > 앱 bundled schema_version  → SKIP (앱 업데이트 필요, CF 폴백)
   b. 앱 version < manifest.min_app_version                → SKIP
   c. manifest.version <= 로컬 version                     → SKIP (이미 최신)
   d. 위 통과                                              → 다운로드 진행
4. 3자산 다운로드(model+tokenizer+labels) → .partial → sha256 검증 → 원자 교체
5. 로컬 version < manifest.min_lcm_version 이면 → SML 비활성, CF only 모드
6. 다음 명령부터 새 LCM 사용
7. 임의 단계 실패 → 직전 정상(N-1) 캐시 → 없으면 bundled → 최종 CF 폴백
```

`fast_path` 의 `_apply`(더 높은 version 만 적용) 로직을 모델용으로 일반화하면 된다(§5).

---

## 4. "최저 / 권장 / 최상" 등급 — 설계 축별 trade-off

각 축에서 **최저(naive·위험)** → **권장(baseline)** → **최상(robust)** 으로 등급화. 라리엔은
**권장을 기본**으로, 보안·치명 경로는 **최상**을 채택한다.

| 설계 축 | 최저 ⚠️ | 권장 ✅(라리엔 기본) | 최상 🏆 |
|---|---|---|---|
| 무결성 검증 | 파일 크기 비교만(현 sherpa) | 자산별 `sha256` | sha256 + **코드 서명**(공개키 검증) |
| 호환 게이트 | 없음(버전만) | `schema_version` 단일 | `schema_version` + `min_app_version` + `min_lcm_version` 3중 |
| fallback | 없음(다운로드 강제) | **bundled 1개** | bundled + **N-1 캐시 롤백** |
| 다운로드 타이밍 | 첫 명령 때 동기(UX 끊김) | 앱 시작 백그라운드 | 백그라운드 + 델타/압축 + 재시도 백오프 |
| 재훈련 동기 강제 | 수동 기억 | CI `sync_ssot.py --check` | CI + `ssot_hash` 런타임 표기 대조 |
| 채널 | 단일 | stable | stable/beta 분리(staging 선검증) |
| 원자 교체 | 파일 직접 덮어쓰기(중단 시 깨짐) | `.partial → rename` | 버전 디렉토리 + 심볼릭 스왑 |

**최악 시나리오 방어 체크**: ① 다운로드 중 앱 종료 → `.partial` 폐기(권장). ② 모델만
교체되고 labels 옛것 → **원자 세트 다운로드**로 차단(§1-3). ③ 잘못 학습된 모델 배포 →
`min_lcm_version` 올려 강제 무효화(최상). ④ 새 action 인데 구 앱 → `schema_version` SKIP +
CF 폴백(절대 silent fail 아님).

---

## 5. 라리엔에 이미 있는 재사용 자산(선례)

| 선례 | 파일 | LCM 이 가져올 것 |
|---|---|---|
| fast_path version 비교 stale-while-revalidate | [lib/services/voice/voice_fast_path.dart](../lib/services/voice/voice_fast_path.dart) | `_apply`(더 높은 version 만)·캐시 로드·String.fromEnvironment URL |
| 큰 onnx 다운로드+크기검증+문서디렉토리 캐시 | [lib/services/voice/sherpa_stt.dart](../lib/services/voice/sherpa_stt.dart) | `_ensureFile`(url→dest, .partial, 진행률) — sha256 추가 |
| 클라 버전 게이트(major*1000+minor) | `kClientCompat`(GAME-SERVER.md §버전 게이트) | `min_app_version`/`schema_version` 판정 패턴 |
| 런타임 labels/headSpecs 주입 | [dart/lib/lcm_classifier.dart](dart/lib/lcm_classifier.dart) `LcmClassifier(labels:, headSpecs:, infer:)` | **OTA 로 받은 labels.json 을 그대로 주입**(이미 설계됨) |

→ `LcmClassifier` 가 이미 labels/headSpecs 를 생성자 주입받으므로, **OTA 자산을 꽂는 데 분류기
코드 변경이 거의 없다.** 다운로드 계층만 새로 만든다.

---

## 6. 재훈련이 필요한가 — 판정표(두 분석 합의)

**재훈련 불필요**(클라 SSOT/서버만 수정): 몬스터 스폰 위치·존 좌표/marker·몬스터 레벨/HP/스폰
수·사냥터 추천 로직·DSL 파라미터 변경. → LCM 은 id 만 출력, 좌표·실행은 클라 책임.

**재훈련 필요**(라벨 공간/표현 변경): 새 action·새 slot·새 메뉴 target·새 DSL 문법·새 존/몬스터
*종류(라벨)* 추가·한국어/STT 커버리지 향상·실사용 로그 오분류 누적.
→ 그중 **새 action/slot(=`schema_version` 상승)** 은 앱 decoder 도 바뀌므로 **Flutter 앱
업데이트도 필요**. 그 외(라벨 추가·표현 보강)는 **모델만 OTA 로 충분**.

재훈련 절차(사람이 원할 때, 자동 합성이라 저비용):
```bash
python scripts/sync_ssot.py        # ① dart SSOT → ssot.json
python -m lcm.gen_data             # ② synth.py 재합성(사람 라벨링 0)
python -m lcm.train --epochs 25    # ③ 학습
python -m lcm.export_onnx          # ④ int8 onnx 출력
# ⑤ versioned 업로드 + manifest 갱신(version/schema_version/sha256)
```

---

## 7. 라리엔 적용 단계 + 차단지점

> **이미 완성**(LCM repo): dart BPE 토큰화·decode·3계층 classify(`LcmClassifier`)·모델/토크나이저
> 아티팩트. **남은 것 = 라리엔 lib 통합 + 서버 배포뿐.**

| 단계 | 작업 | 자율/차단 |
|---|---|---|
| S1 | `labels.json` 분리 export(`export_golden.py` 확장: golden 에서 labels/head_specs/threshold/pad_len 추출) | ✅ AI 자율(LCM repo 내) |
| S2 | manifest.json 생성 스크립트(version/schema_version/sha256/min_* 채움) | ✅ AI 자율 |
| S3 | **onnxruntime Dart 바인딩 도입**(`InferFn` 채우기) | 🛑 **차단지점** — 비공식 패키지(CLAUDE.md §Flame 공식 최우선). 사용자 승인 필요 |
| S4 | lib/ OTA 다운로드 계층(`lcm_ota.dart`: manifest fetch·게이트·sha256·N-1 캐시·`LcmClassifier` 주입) | 🛑 **차단지점** — lib 변경 + flutter 빌드 + **DTD 시각 검증 의무** |
| S5 | `voice_command_sheet.dart` fast-path 실패 경로에 LCM 2차 삽입, 실패 시 기존 CF 폴백 | 🛑 차단지점(S4 동반) |
| S6 | 모델·자산·manifest 를 `laryen.com/models/lcm/` 업로드(stale-while-revalidate) | ✅ staging 자율 / production 사람 |
| S7 | CI 가드: `sync_ssot.py --check` 를 pre-push hook(F-046 verify.sh)에 합류 | ✅ AI 자율 |

**핵심 차단지점 = S3(비공식 onnxruntime 패키지 도입)**. 이것만 사용자가 승인하면 S4~S5 가
열린다. (sherpa_onnx 가 ORT 세션을 재노출하면 추가 패키지 0일 수 있어 우선 조사 가능.)

---

## 8. 한 줄 요약

Flutter 빌드에 LCM 을 임베드하지 말고, **bundled fallback + manifest 기반 원격 최신 모델 2단
구조**로 간다. manifest 는 `version`(신선도) · `schema_version`/`min_app_version`(decoder 호환) ·
`min_lcm_version`(강제 무효화) · 자산별 `sha256` 를 담고, **model+tokenizer+labels 를 원자 세트**
로 받는다. 다운로드 실패·구버전·호환 불가는 모두 **CF 폴백**으로 안전하게 흡수된다. 라리엔은
fast_path·sherpa·`LcmClassifier` 주입 설계를 재사용하므로 통합 비용이 낮고, 유일한 선결
차단지점은 **onnxruntime Dart 바인딩 승인(S3)** 이다.
