//! 真实数据验证（依赖 data/ 下生成的文件，标记 ignore 防止普通跑测试时误用）。

use cnt_lm::CntLm;

#[test]
#[ignore = "依赖 data/ 下生成的真实数据文件（不入库）"]
fn real_lm_queries() {
    let lm = CntLm::open("../../data/lm.cntl").unwrap();
    for (w1, w2) in [("没", "命"), ("命", "其"), ("其", "妙"), ("大", "字"), ("莫", "名"), ("名", "其"), ("死", "后")] {
        let _ = (w1, w2);
    }
    for w in ["时候", "打字", "大字", "莫名其妙", "没", "命", "其", "妙", "大", "莫", "名", "字"] {
        eprintln!("unigram({w}) = {:?}", lm.unigram(w));
    }
    for (w1, w2) in [("时", "候"), ("是", "候"), ("是", "后"), ("时", "后"), ("没", "命"), ("命", "其"), ("其", "妙"), ("大", "字"), ("莫", "名"), ("名", "其"), ("死", "后")] {
        let u1 = lm.unigram(w1);
        let u2 = lm.unigram(w2);
        let bg = lm.bigram(w1, w2);
        eprintln!("unigram({w1})={u1:?} unigram({w2})={u2:?} bigram({w1},{w2})={bg:?}");
    }
}
