# -*-: coding: utf-8 -*-

"""
pip install diff-match-patch

1.0.2 Increase Diff_Timeout
1.0.3 Ignore source text newlines
1.0.4 Optimize line ends
"""

from pprint import pp
from pathlib import Path
from diff_match_patch import diff_match_patch

_version = '1.0.4'

DIFF_TIMEOUT = 320
DIFF_EDITCOST = 4


def print_diffs(diffs):
    for line_num, diff in diffs:
        print(f"\n🔹 第 {line_num} 行不同：")
        for op, data in diff:
            if op == 0:
                print(f"  相同: {data}")
            elif op == -1:
                print(f"  ❌ 删除: {data}")
            elif op == 1:
                print(f"  ✅ 插入: {data}")


def diff_by_lines_unlimited(text1, text2) -> list[list] :
    dmp = diff_match_patch()

    # 处理长文本时增加超时限制
    dmp.Diff_Timeout = DIFF_TIMEOUT
    
    # 可调节替换与插入/删除的偏好
    dmp.Diff_EditCost = DIFF_EDITCOST

    diff = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diff)
    return diff


def is_break(c: str) -> bool :
    if c[-1].isspace() :
        return True
    elif c[-1] in ('(', '“', '—') :
        return True
    return False


def adjust_newline_text(content: str) -> str :
    starting_punctuations = ('(', '“', '‘')
    ending_punctuations = (')', '”', '’')
    for p in starting_punctuations :
        # Move p to next line start
        content = content.replace(p + '\n', '\n' + p)
    for p in ending_punctuations :
        # Move p to previous line end
        content = content.replace('\n' + p, p + '\n')
    return content


def generate_diff_file(original_file: str|Path, transcribed_file: str|Path, verbose=False) :
    original = Path(original_file)
    transcribed = Path(transcribed_file)
    assert original.exists() and transcribed.exists(), 'File not found!'
    print('Comparing', original.name, 'VS', transcribed.name)
    with open(original, 'r', encoding='utf-8') as f1, open(transcribed, 'r', encoding='utf-8') as f2 :
        text1 = f1.read().replace('\n', ' ')
        text2 = f2.read()

    diffs = diff_by_lines_unlimited(text1, text2)
    if verbose :
        pp(diffs)
    text_buffer = []
    previous_op = 0
    previous_content = ''
    for d in diffs:
        match d :
            # 保持原文不变
            case (0, content) | (-1, content) :
                text_buffer.append(content)
            # 转录文本新增换行，位置在句尾，直接插入
            case (1, content) if content.endswith('\n') :
                text_buffer.append('\n')
                # 句中换行加标记 &
                if not is_break(previous_content) and args.mark :
                    text_buffer.append('&')
            # 转录文本新增换行，位置在一个单词前，插入前一位
            case (1, content) if content.startswith('\n') and previous_op == -1 :
                text_buffer.insert(-1, '\n')
            # 转录文本新增换行，转录文本多一个单词，可以忽略，直接插入
            case (1, content) if content.startswith('\n') and previous_op == 0 :
                text_buffer.append('\n')
        previous_op, previous_content = d
    new_line = original.with_stem(original.stem + '_newline')
    # adjust_newline_text 调整行首或行尾的特殊标点符号
    new_line.write_text(adjust_newline_text(''.join(text_buffer)), encoding='utf-8')


def main(args) :
    generate_diff_file(transcribed_file=args.transcribed, original_file=args.original, verbose=args.verbose)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description='根据转录生成的断行稿给原文断行')
    parser.add_argument('-v', '--version', action='version', version=f"version: {_version}", help='显示版本并退出')
    parser.add_argument('-t', '--transcribed', required=True, help='转录文本')
    parser.add_argument('-o', '--original', required=True, help='原文本')
    parser.add_argument('-a', '--mark', action='store_true', help='加入断行标记 &')
    parser.add_argument('--verbose', action='store_true', help='开启 verbose 模式')
    args = parser.parse_args()
    main(args)
