"""다국어 동등 지원 — 한국어·영어·중국어·일본어를 *동일 우선순위* 로(2026-06-24 사용자 지시).

**왜 backbone 전제인가**: from-scratch 인코더(vocab 4000, 라리엔 합성 코퍼스)는 중/일 어휘를
거의 못 다룬다. multilingual-e5-small backbone 은 한·영·중·일을 사전학습해 의미 임베딩은
동등하나, *라리엔 의도 매핑* 을 4개 언어 모두 정확히 하려면 학습 데이터에 4개 언어를 동등
비율로 넣어야 한다. 본 모듈이 answer_intent 질문 + 핵심 명령을 4개 언어로 동등 생성한다.

고유명사(지역·몬스터)는 한국어 게임 용어라 _gen_move/_gen_hunt(기존)가 한/영으로 다루고,
여기서는 *언어 구조에 의존하는* 의도/명령을 4개 언어로 커버한다(지역 없는 명령 + 질문).
"""
from __future__ import annotations

LANGS = ("ko", "en", "zh", "ja")

# answer_intent 질문 — 4개 언어 동등(고유명사 적어 자연 번역). 답변은 게임지식 OTA answers.
I18N_QUESTION: dict[str, dict[str, list[str]]] = {
    "query_potion_effect": {
        "ko": ["물약 효과 뭐야", "포션 효과 알려줘", "물약 종류 뭐 있어", "어떤 물약이 있어", "물약 설명해줘"],
        "en": ["what do potions do", "potion effects", "what potions are there", "tell me about potions", "explain the potions"],
        "zh": ["药水有什么效果", "药水的作用是什么", "有哪些药水", "药水种类有哪些", "介绍一下药水"],
        "ja": ["ポーションの効果は", "薬の効果を教えて", "どんなポーションがある", "ポーションの種類は", "ポーションについて教えて"],
    },
    "query_gear_set_effect": {
        "ko": ["세트 효과 뭐야", "세트 아이템 효과", "장비 세트 알려줘", "어떤 세트가 강해", "세트 능력치 뭐야"],
        "en": ["gear set effect", "what do gear sets do", "set item bonus", "which set is strong", "explain gear sets"],
        "zh": ["套装效果是什么", "装备套装的作用", "套装加成是什么", "哪个套装最强", "介绍一下套装"],
        "ja": ["セット効果は", "装備セットの効果", "セットボーナスは", "どのセットが強い", "セットについて教えて"],
    },
    "query_level_progression": {
        "ko": ["레벨업 어떻게 해", "경험치 어떻게 벌어", "어떻게 강해져", "최대 레벨 몇이야", "빨리 레벨 올리는 법"],
        "en": ["how to level up", "how to gain exp", "how to get stronger", "what is the max level", "fastest way to level"],
        "zh": ["怎么升级", "怎么获得经验", "怎么变强", "最高等级是多少", "快速升级的方法"],
        "ja": ["レベルアップはどうやって", "経験値の稼ぎ方", "どうやって強くなる", "最大レベルはいくつ", "早くレベルを上げる方法"],
    },
    "query_party_info": {
        "ko": ["파티 어떻게 만들어", "파티 뭐야", "같이 사냥 어떻게 해", "파티 인원 몇명", "파티 경험치 어떻게"],
        "en": ["how to make a party", "what is a party", "how does party work", "party member limit", "how is party exp shared"],
        "zh": ["怎么组队", "组队是什么", "怎么一起打怪", "队伍人数上限", "组队经验怎么分"],
        "ja": ["パーティーの作り方", "パーティーとは", "一緒に狩りするには", "パーティーの人数は", "パーティー経験値の分配"],
    },
    "query_trade_info": {
        "ko": ["교환 어떻게 해", "거래 어떻게 해", "교환 뭐야", "아이템 교환 방법", "물약 교환 어떻게"],
        "en": ["how to trade", "how does trade work", "what is trading", "how to exchange items", "how to trade potions"],
        "zh": ["怎么交易", "交易怎么进行", "交易是什么", "怎么交换道具", "怎么交换药水"],
        "ja": ["トレードはどうやって", "取引のやり方", "交換とは", "アイテムの交換方法", "ポーションの交換"],
    },
    "query_world_lore": {
        "ko": ["세계관이 뭐야", "이 게임 스토리", "옵시디언이 뭐야", "넥서스가 뭐야", "게임 배경 알려줘"],
        "en": ["what is the world setting", "this game's story", "what is obsidian", "what is nexus", "tell me the lore"],
        "zh": ["世界观是什么", "这游戏的故事", "黑曜石是什么", "Nexus是什么", "介绍游戏背景"],
        "ja": ["世界観は何", "このゲームのストーリー", "オブシディアンとは", "ネクサスとは", "ゲームの背景を教えて"],
    },
    "query_help_controls": {
        "ko": ["어떻게 조작해", "어떻게 플레이해", "조작법 알려줘", "어떻게 시작해", "게임 어떻게 해"],
        "en": ["how do i play", "how to control", "tell me the controls", "how do i start", "how does this game work"],
        "zh": ["怎么操作", "怎么玩", "操作方法是什么", "怎么开始", "这游戏怎么玩"],
        "ja": ["操作方法は", "どうやって遊ぶ", "操作を教えて", "どうやって始める", "このゲームのやり方"],
    },
    "query_player_level": {
        "ko": ["내 레벨 몇이야", "나 몇 레벨", "현재 레벨 알려줘", "내 레벨 확인"],
        "en": ["what level am i", "my level", "what is my current level", "check my level"],
        "zh": ["我是几级", "我的等级", "我现在多少级", "查看我的等级"],
        "ja": ["私のレベルは", "今何レベル", "現在のレベルは", "レベルを確認"],
    },
    "query_recommended_hunt_zone": {
        "ko": ["어디서 사냥하면 좋아", "내 레벨 사냥터 어디", "추천 사냥터", "어디 가서 사냥해"],
        "en": ["where should i hunt", "best hunting spot for my level", "recommended hunting ground", "where to hunt"],
        "zh": ["在哪里打怪好", "适合我等级的狩猎场", "推荐的狩猎地点", "去哪里打怪"],
        "ja": ["どこで狩りすればいい", "私のレベルの狩場", "おすすめの狩場", "どこで狩りする"],
    },
    "query_monster_info": {
        "ko": ["제일 센 몬스터 뭐야", "몬스터 종류 뭐 있어", "어떤 몬스터가 강해"],
        "en": ["what is the strongest monster", "what monsters are there", "which monster is strong"],
        "zh": ["最强的怪物是什么", "有哪些怪物", "哪个怪物厉害"],
        "ja": ["一番強いモンスターは", "どんなモンスターがいる", "どのモンスターが強い"],
    },
    # 메타 질문(도우미 정체성) — "너 이름 뭐야"·"어떤 AI" 류. answers 즉답(DeepSeek 안 감).
    "query_assistant_identity": {
        "ko": ["너 이름 뭐야", "넌 누구야", "어떤 AI 야", "누가 만들었어", "넌 뭐로 만들어졌어",
               "너 뭐야", "이름이 뭐야", "정체가 뭐야", "넌 무슨 인공지능이야"],
        "en": ["what is your name", "who are you", "what ai are you", "who made you",
               "what are you built with", "what are you", "tell me about yourself"],
        "zh": ["你叫什么名字", "你是谁", "你是什么AI", "谁创造了你", "你是用什么做的", "你是什么"],
        "ja": ["名前は何", "あなたは誰", "どんなAIなの", "誰が作ったの", "何で作られてるの", "君は何"],
    },
    "query_assistant_capabilities": {
        "ko": ["뭘 할 수 있어", "무엇을 도와줄 수 있어", "어떤 기능이 있어", "뭐 할 수 있는데",
               "무슨 명령 할 수 있어", "널 어떻게 써"],
        "en": ["what can you do", "what can you help with", "what are your features",
               "what commands can i say", "how do i use you"],
        "zh": ["你能做什么", "你能帮我做什么", "你有什么功能", "我能说什么命令"],
        "ja": ["何ができる", "何を手伝える", "どんな機能がある", "どんな命令ができる"],
    },
    "query_voice_privacy": {
        "ko": ["내 목소리 서버로 가", "로컬에서 처리해", "음성이 저장돼", "내 말 어디로 가",
               "녹음이 서버로 가", "개인정보 안전해"],
        "en": ["does my voice go to the server", "is it processed locally", "is my voice stored",
               "where does my voice go", "is my data safe"],
        "zh": ["我的声音会上传服务器吗", "是本地处理吗", "我的语音会被保存吗", "数据安全吗"],
        "ja": ["私の声はサーバーに送られる", "ローカルで処理される", "音声は保存される", "データは安全"],
    },
}

# 지역 없는 핵심 명령 — 4개 언어 동등. (action, mode, target) 라벨.
I18N_COMMAND: dict[str, dict[str, list[str]]] = {
    "potion": {  # {action: potion}
        "ko": ["물약 먹어", "포션 마셔", "회복해", "피 채워", "물약 써"],
        "en": ["use potion", "drink potion", "heal", "use hp potion", "take a potion"],
        "zh": ["使用药水", "喝药水", "恢复生命", "加血", "用药水"],
        "ja": ["ポーション使って", "薬を飲んで", "回復して", "HP回復", "ポーションを使う"],
    },
    "hunt": {  # {action: hunt}
        "ko": ["사냥해", "사냥하자", "몬스터 잡아", "사냥 시작", "사냥 좀 하자", "사냥 좀 하자고",
               "사냥이나 하자", "사냥하러 가자", "몹 잡자", "사냥 좀 해줘"],
        "en": ["hunt", "start hunting", "kill monsters", "let's hunt", "let's go hunting", "go kill some mobs"],
        "zh": ["打怪", "开始打怪", "去刷怪", "我们去打怪", "去打怪吧", "刷怪去"],
        "ja": ["狩りして", "狩りを始めて", "モンスター倒して", "狩りしよう", "狩りに行こう", "モンスター狩ろう"],
    },
    "stop": {  # {action: stop}
        "ko": ["멈춰", "그만", "정지", "스톱"],
        "en": ["stop", "halt", "stop it", "cancel"],
        "zh": ["停下", "停止", "别动", "停"],
        "ja": ["止まって", "ストップ", "停止", "やめて"],
    },
    "open_menu": {  # {action: open_menu}
        "ko": ["메뉴 열어", "메뉴 보여줘", "메뉴", "메뉴 띄워", "메뉴 좀 열어", "메뉴 열어줘", "메뉴 열기"],
        "en": ["open menu", "show menu", "menu", "open the menu", "bring up menu"],
        "zh": ["打开菜单", "显示菜单", "菜单", "把菜单打开", "调出菜单"],
        "ja": ["メニュー開いて", "メニューを見せて", "メニュー", "メニューを出して", "メニューを開く"],
    },
    "auto_hunt": {  # {action: auto_combat, mode: auto_hunt}
        "ko": ["자동사냥 켜", "오토 켜", "자동전투 시작", "자동사냥 시작해"],
        "en": ["turn on auto hunt", "enable auto combat", "start auto hunt", "auto hunt on"],
        "zh": ["开启自动狩猎", "打开自动战斗", "开始自动打怪", "自动狩猎开"],
        "ja": ["オート狩り オン", "自動戦闘を開始", "オートハント開始", "自動狩りをつけて"],
    },
    "auto_off": {  # {action: auto_combat, mode: off}
        "ko": ["자동사냥 꺼", "오토 꺼", "자동전투 중지", "자동사냥 그만"],
        "en": ["turn off auto hunt", "disable auto combat", "stop auto hunt", "auto hunt off"],
        "zh": ["关闭自动狩猎", "停止自动战斗", "关掉自动打怪", "自动狩猎关"],
        "ja": ["オート狩り オフ", "自動戦闘を停止", "オートハント停止", "自動狩りを切って"],
    },
}

# 🛑 맥락지시어 — 무엇을 가리키는지 모름(거기/그거/아까 그곳). 임의 실행 금지 = ambiguous
# → route=clarify(되묻기). 4개 언어. backbone 이 의미로 일반화하도록 충분히 다양하게.
I18N_AMBIGUOUS: dict[str, list[str]] = {
    "ko": ["거기로 가", "그거 잡아", "그쪽으로 가줘", "아까 말한 곳으로 가", "저번에 갔던 데로 가자",
           "그 위치로 이동", "거기서 사냥해", "아까 그 몬스터 잡아", "방금 그거 공격해",
           "거기 있는 거 잡아", "그 사냥터로 가", "아까 거기로", "그 근처로 이동해", "저기 그거 잡아"],
    "en": ["go there", "get that one", "go that way", "to the place i mentioned", "where we went before",
           "move to that spot", "hunt over there", "attack that monster from before", "kill that thing",
           "go to that hunting ground", "back to that place", "move near there"],
    "zh": ["去那里", "打那个", "往那边走", "去之前说的地方", "去上次那里", "移动到那个位置",
           "在那里打怪", "攻击刚才那个怪", "打那个东西", "去那个狩猎场", "回到那个地方"],
    "ja": ["そこに行って", "あれを倒して", "あっちに行って", "さっき言った場所へ", "前に行ったところへ",
           "あの場所に移動", "そこで狩りして", "さっきのモンスターを倒して", "あれを攻撃",
           "あの狩場に行って", "あの場所に戻って"],
}

_COMMAND_INTENT = {
    "potion": {"action": "potion"},
    "hunt": {"action": "hunt"},
    "stop": {"action": "stop"},
    "open_menu": {"action": "open_menu"},
    "auto_hunt": {"action": "auto_combat", "mode": "auto_hunt"},
    "auto_off": {"action": "auto_combat", "mode": "off"},
}


def gen_i18n() -> list[tuple[str, dict]]:
    """한·영·중·일 4개 언어 동등 발화. (text, intent) 리스트."""
    out: list[tuple[str, dict]] = []
    for intent_name, by_lang in I18N_QUESTION.items():
        for _lang, texts in by_lang.items():
            for t in texts:
                out.append((t, {"action": "unknown", "semantic_type": "question",
                                "answer_intent": intent_name}))
    for key, by_lang in I18N_COMMAND.items():
        base = _COMMAND_INTENT[key]
        for _lang, texts in by_lang.items():
            for t in texts:
                out.append((t, dict(base)))
    # 맥락지시어 → ambiguous(action=unknown, 실행 금지 → clarify). 4개 언어.
    for _lang, texts in I18N_AMBIGUOUS.items():
        for t in texts:
            out.append((t, {"action": "unknown", "semantic_type": "ambiguous"}))
    return out


def lang_stats() -> dict[str, int]:
    """언어별 생성 발화 수(균형 확인용)."""
    c = {l: 0 for l in LANGS}
    for by_lang in list(I18N_QUESTION.values()) + list(I18N_COMMAND.values()):
        for lang, texts in by_lang.items():
            c[lang] += len(texts)
    return c
