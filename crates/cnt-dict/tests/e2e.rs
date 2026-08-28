//! 端到端：文本词表 → 编译二进制 → mmap 打开 → 查询。

use cnt_dict::writer;
use cnt_dict::{MmapDict, PinyinModel};

#[test]
fn build_then_mmap_then_query() {
    let dir = std::env::temp_dir().join(format!("cnt-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let dict_path = dir.join("dict.cntd");
    let user_path = dir.join("user.dict");

    let pairs = vec![
        ("nihao".to_string(), "你好".to_string(), 1000),
        ("nihaoma".to_string(), "你好吗".to_string(), 950),
        ("ni".to_string(), "你".to_string(), 900),
        ("ni".to_string(), "泥".to_string(), 100),
    ];
    writer::write_to_file(&pairs, &dict_path).unwrap();

    let d = MmapDict::open(&dict_path).unwrap();
    assert_eq!(d.entry_count(), 3);
    assert_eq!(d.key_at(0), Some("ni"));
    assert_eq!(d.exact("ni")[0].word, "你");

    let m = PinyinModel::open(&dict_path, &user_path).unwrap();
    let q = m.query("ni");
    // 精确读音优先：前缀词按 PREFIX_DISCOUNT 折价（你好 1000/4=250），
    // 不再用全额词频压过同拼音的精确词 你(900)
    assert_eq!(q[0], "你");
    assert_eq!(q[1], "你好");
    assert_eq!(q[2], "你好吗");

    m.bump("ni", "你");
    m.bump("ni", "你");
    m.flush_user().unwrap();
    let m2 = PinyinModel::open(&dict_path, &user_path).unwrap();
    let q2 = m2.query("ni");
    assert_eq!(q2[0], "你"); // 用户调频后挤到第一

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn user_words_appear_at_partial_input() {
    // 学过的复合词（chijiuhua/持久化）在输入到一半（chijiuh）时也应出现
    let dir = std::env::temp_dir().join(format!("cnt-e2e-prefix-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let dict_path = dir.join("dict.cntd");
    let user_path = dir.join("user.dict");
    writer::write_to_file(&[("chi".to_string(), "持".to_string(), 100)], &dict_path).unwrap();
    let m = PinyinModel::open(&dict_path, &user_path).unwrap();
    // 学习复合词
    m.bump("chijiuhua", "持久化");
    m.bump("chijiuhua", "持久化");
    // 部分输入时通过前缀补全出现
    let q = m.query("chijiuh");
    assert!(
        q.contains(&"持久化".to_string()),
        "user word should appear at partial input: {q:?}"
    );
    // 完整输入时也出现且排前
    let q2 = m.query("chijiuhua");
    assert_eq!(q2.first().map(String::as_str), Some("持久化"));
    let _ = std::fs::remove_dir_all(&dir);
}
