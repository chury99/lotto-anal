import os
import sys
import pandas as pd


# noinspection PyUnresolvedReferences,PyProtectedMember,PyPep8Naming
def 번호선정로직_2개번호_최빈수(dic_정보):
    """ 계산된 확률값을 바탕으로 6개 번호 5 세트 선택 - 2개 번호 상위 30개 중 최빈수 선택 """
    # function 이름 찾기
    func_name = sys._getframe(0).f_code.co_name
    s_전략명 = func_name

    # 확률값 가져오기
    df_확률_1개 = dic_정보['df_확률_1개']
    df_확률_2개 = dic_정보['df_확률_2개']
    df_확률_6개 = dic_정보['df_확률_6개']

    # 2개 확률 중 상위 30개 번호 최빈수 찾기
    df_30개 = df_확률_2개[:30].copy()
    df_30개['no1'] = df_30개['no'].apply(lambda x: int(x[2:4]))
    df_30개['no2'] = df_30개['no'].apply(lambda x: int(x[5:7]))
    li_30개 = list(df_30개['no1'].values) + list(df_30개['no2'].values)
    sri_카운트 = pd.Series(li_30개).value_counts()
    n_최빈번호 = sri_카운트.index[0]

    # 2개 번호 골라내기
    ary_2개번호_30개 = df_30개.loc[:, ['no1', 'no2']].values
    li_2개번호_전체 = [list(ary_2개번호) for ary_2개번호 in ary_2개번호_30개 if n_최빈번호 in ary_2개번호]

    # 6개 번호조합에서 2개 번호 포함된 항목 골라내기
    ary_6개번호_전체 = df_확률_6개.loc[:, 'no1': 'no6'].values
    li_6개번호_전체 = [list(ary_6개번호) for ary_6개번호 in ary_6개번호_전체 if n_최빈번호 in ary_6개번호]
    li_6개번호_5개세트 = []
    for li_6개번호 in li_6개번호_전체:
        # 2개 번호 포함 여부 확인
        for li_2개번호 in li_2개번호_전체:
            if (li_2개번호[0] in li_6개번호) and (li_2개번호[1] in li_6개번호):
                li_6개번호_5개세트.append(li_6개번호)
                li_2개번호_전체.remove(li_2개번호)
                break

        if len(li_6개번호_5개세트) >= 5:
            break

    # df 변환
    li_컬럼명 = [f'no{n + 1}' for n in range(6)]
    df_6개번호_5개세트 = pd.DataFrame(li_6개번호_5개세트, columns=li_컬럼명)

    return s_전략명, df_6개번호_5개세트


# noinspection PyUnresolvedReferences,PyProtectedMember,PyPep8Naming
def 번호선정로직_2개번호_최빈수_중복제외(dic_정보):
    """ 계산된 확률값을 바탕으로 6개 번호 5 세트 선택 - 2개 번호 상위 30개 중 최빈수 선택 """
    # function 이름 찾기
    func_name = sys._getframe(0).f_code.co_name
    s_전략명 = func_name

    # 확률값 가져오기
    df_확률_1개 = dic_정보['df_확률_1개']
    df_확률_2개 = dic_정보['df_확률_2개']
    df_확률_6개 = dic_정보['df_확률_6개']

    # 2개 확률 중 상위 30개 번호 최빈수 찾기
    df_30개 = df_확률_2개[:30].copy()
    df_30개['no1'] = df_30개['no'].apply(lambda x: int(x[2:4]))
    df_30개['no2'] = df_30개['no'].apply(lambda x: int(x[5:7]))
    li_30개 = list(df_30개['no1'].values) + list(df_30개['no2'].values)
    sri_카운트 = pd.Series(li_30개).value_counts()
    n_최빈번호 = sri_카운트.index[0]

    # 2개 번호 골라내기
    ary_2개번호_30개 = df_30개.loc[:, ['no1', 'no2']].values
    li_2개번호_전체 = [list(ary_2개번호) for ary_2개번호 in ary_2개번호_30개 if n_최빈번호 in ary_2개번호]
    li_2번째번호 = []
    for li_2개번호 in li_2개번호_전체:
        for n_번호 in li_2개번호:
            if n_번호 != n_최빈번호:
                li_2번째번호.append(n_번호)

    # 6개 번호조합에서 2개 번호 포함된 항목 골라내기
    ary_6개번호_전체 = df_확률_6개.loc[:, 'no1': 'no6'].values
    li_6개번호_전체 = [list(ary_6개번호) for ary_6개번호 in ary_6개번호_전체 if n_최빈번호 in ary_6개번호]
    li_6개번호_5개세트 = []
    li_2번째번호_지운거 = []
    for li_6개번호 in li_6개번호_전체:
        # 2번째 번호 포함 여부 확인
        for n_2번째번호 in li_2번째번호:
            if n_2번째번호 in li_6개번호:
                # 사용한 번호 포함 여부 확인
                li_사용한번호확인 = [1 if n_번호 in li_2번째번호_지운거 else 0 for n_번호 in li_6개번호]
                if sum(li_사용한번호확인) > 0:
                    break
                # 번호 추출
                li_6개번호_5개세트.append(li_6개번호)
                li_2번째번호.remove(n_2번째번호)
                li_2번째번호_지운거.append(n_2번째번호)
                break

        if len(li_6개번호_5개세트) >= 5:
            break

    # df 변환
    li_컬럼명 = [f'no{n + 1}' for n in range(6)]
    df_6개번호_5개세트 = pd.DataFrame(li_6개번호_5개세트, columns=li_컬럼명)

    return s_전략명, df_6개번호_5개세트


# noinspection PyUnresolvedReferences,PyProtectedMember,PyPep8Naming
def 번호선정로직_2개번호_따라가기(dic_정보):
    """ 1개 번호 선정 후 2개 번호 중 확률 높은 거 기준으로 따라가기 """
    # function 이름 찾기
    func_name = sys._getframe(0).f_code.co_name
    s_전략명 = func_name

    # 확률값 가져오기
    df_확률_1개 = dic_정보['df_확률_1개']
    df_확률_2개 = dic_정보['df_확률_2개']

    # 예측번호 1개 확인
    ary_예측번호_1개_s = df_확률_1개[:5]['no'].values
    li_예측번호_1개_n = [int(s[2:]) for s in ary_예측번호_1개_s]

    # 예측번호 2개 숫자 분리
    df_확률_2개['no1'] = df_확률_2개['no'].apply(lambda x: int(x[2:4]))
    df_확률_2개['no2'] = df_확률_2개['no'].apply(lambda x: int(x[5:7]))

    # 1개확률 번호별 번호조합 생성
    li_li_번호조합 = []
    for n_번호1 in li_예측번호_1개_n:
        li_번호조합 = [n_번호1]
        # 2개번호 중 최고확률 찾기
        for i in range(5):
            n_번호_앞 = li_번호조합[-1]
            li_df_2개번호 = [df_확률_2개[df_확률_2개['no1'] == n_번호_앞], df_확률_2개[df_확률_2개['no2'] == n_번호_앞]]
            df_2개번호 = pd.concat(li_df_2개번호, axis=0).drop_duplicates().sort_values('prob_1', ascending=False)
            for j in range(5):
                ary_번호_뒤 = df_2개번호.loc[:, ['no1', 'no2']].values[j]
                n_번호_뒤 = [int(n) for n in ary_번호_뒤 if int(n) is not n_번호_앞][0]
                if n_번호_뒤 not in li_번호조합:
                    li_번호조합.append(n_번호_뒤)
                    break
                else:
                    continue
        li_번호조합.sort()
        li_li_번호조합.append(li_번호조합)

    # df 변환
    li_컬럼명 = [f'no{n + 1}' for n in range(6)]
    df_6개번호_5개세트 = pd.DataFrame(li_li_번호조합, columns=li_컬럼명)

    return s_전략명, df_6개번호_5개세트


# noinspection PyUnresolvedReferences,PyProtectedMember,PyPep8Naming
def 번호선정로직_2개번호_따라가기_확률반영(dic_정보):
    """ 1개 번호 선정 후 2개 번호 중 확률 높은 거 기준으로 따라가기 """
    # function 이름 찾기
    func_name = sys._getframe(0).f_code.co_name
    s_전략명 = func_name

    # 확률값 가져오기
    df_확률_1개 = dic_정보['df_확률_1개']
    df_확률_2개 = dic_정보['df_확률_2개']

    # 예측번호 1개 확인
    ary_예측번호_1개_s = df_확률_1개[:20]['no'].values
    li_예측번호_1개_n = [int(s[2:]) for s in ary_예측번호_1개_s]

    # 예측번호 2개 숫자 분리
    df_확률_2개['no1'] = df_확률_2개['no'].apply(lambda x: int(x[2:4]))
    df_확률_2개['no2'] = df_확률_2개['no'].apply(lambda x: int(x[5:7]))
    dic_확률_2개 = df_확률_2개.set_index('no').to_dict()['prob_1']

    # 1개확률 번호별 번호조합 생성
    li_li_번호조합 = []
    for n_번호1 in li_예측번호_1개_n:
        li_번호조합 = [n_번호1]

        # 2번째 번호 찾기
        li_df_2개번호 = [df_확률_2개[df_확률_2개['no1'] == n_번호1], df_확률_2개[df_확률_2개['no2'] == n_번호1]]
        df_2개번호 = pd.concat(li_df_2개번호, axis=0).drop_duplicates().sort_values('prob_1', ascending=False)
        ary_번호2 = df_2개번호.loc[:, ['no1', 'no2']].values[0]
        n_번호2 = [int(n) for n in ary_번호2 if int(n) is not n_번호1][0]
        li_번호조합.append(n_번호2)

        # 3번째 번호 찾기
        li_번호조합.sort()
        li_번호3 = [n for n in [n + 1 for n in range(45)]
                  if (int(n) is not n_번호1) and (int(n) is not n_번호2)]

        n_확률_12 = dic_확률_2개[f'no{li_번호조합[0]:02}|{li_번호조합[1]:02}']
        li_번호조합_13 = [[n_번호1, n_번호3] if n_번호1 < n_번호3 else [n_번호3, n_번호1] for n_번호3 in li_번호3]
        li_번호조합_13 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_13]
        li_번호조합_23 = [[n_번호2, n_번호3] if n_번호2 < n_번호3 else [n_번호3, n_번호2] for n_번호3 in li_번호3]
        li_번호조합_23 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_23]

        df_번호3_확률 = pd.DataFrame()
        df_번호3_확률['no'] = li_번호조합_23
        df_번호3_확률['prob_12'] = n_확률_12
        df_번호3_확률['prob_13'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_13]
        df_번호3_확률['prob_23'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_23]
        ary_확률 = df_번호3_확률.loc[:, 'prob_12': 'prob_23'].values
        df_번호3_확률['prob_total'] = [ary.prod() for ary in ary_확률]
        df_번호3_확률 = df_번호3_확률.sort_values('prob_total', ascending=False)

        n_번호3 = [int(s) for s in df_번호3_확률['no'].values[0].replace('no', '').split('|')
                 if int(s) is not n_번호2][0]
        li_번호조합.append(n_번호3)

        # 4번째 번호 찾기
        li_번호조합.sort()
        li_번호4 = [n for n in [n + 1 for n in range(45)]
                  if (int(n) is not n_번호1) and (int(n) is not n_번호2) and (int(n) is not n_번호3)]

        n_확률_123 = df_번호3_확률['prob_total'].values[0]
        li_번호조합_14 = [[n_번호1, n_번호4] if n_번호1 < n_번호4 else [n_번호4, n_번호1] for n_번호4 in li_번호4]
        li_번호조합_14 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_14]
        li_번호조합_24 = [[n_번호2, n_번호4] if n_번호2 < n_번호4 else [n_번호4, n_번호2] for n_번호4 in li_번호4]
        li_번호조합_24 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_24]
        li_번호조합_34 = [[n_번호3, n_번호4] if n_번호3 < n_번호4 else [n_번호4, n_번호3] for n_번호4 in li_번호4]
        li_번호조합_34 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_34]

        df_번호4_확률 = pd.DataFrame()
        df_번호4_확률['no'] = li_번호조합_34
        df_번호4_확률['prob_123'] = n_확률_123
        df_번호4_확률['prob_14'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_14]
        df_번호4_확률['prob_24'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_24]
        df_번호4_확률['prob_34'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_34]
        ary_확률 = df_번호4_확률.loc[:, 'prob_123': 'prob_34'].values
        df_번호4_확률['prob_total'] = [ary.prod() for ary in ary_확률]
        df_번호4_확률 = df_번호4_확률.sort_values('prob_total', ascending=False)

        n_번호4 = [int(s) for s in df_번호4_확률['no'].values[0].replace('no', '').split('|')
                 if int(s) is not n_번호3][0]
        li_번호조합.append(n_번호4)

        # 5번째 번호 찾기
        li_번호조합.sort()
        li_번호5 = [n for n in [n + 1 for n in range(45)]
                  if (int(n) is not n_번호1) and (int(n) is not n_번호2) and (int(n) is not n_번호3)
                  and (int(n) is not n_번호4)]

        n_확률_1234 = df_번호4_확률['prob_total'].values[0]
        li_번호조합_15 = [[n_번호1, n_번호5] if n_번호1 < n_번호5 else [n_번호5, n_번호1] for n_번호5 in li_번호5]
        li_번호조합_15 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_15]
        li_번호조합_25 = [[n_번호2, n_번호5] if n_번호2 < n_번호5 else [n_번호5, n_번호2] for n_번호5 in li_번호5]
        li_번호조합_25 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_25]
        li_번호조합_35 = [[n_번호3, n_번호5] if n_번호3 < n_번호5 else [n_번호5, n_번호3] for n_번호5 in li_번호5]
        li_번호조합_35 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_35]
        li_번호조합_45 = [[n_번호4, n_번호5] if n_번호4 < n_번호5 else [n_번호5, n_번호4] for n_번호5 in li_번호5]
        li_번호조합_45 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_45]

        df_번호5_확률 = pd.DataFrame()
        df_번호5_확률['no'] = li_번호조합_45
        df_번호5_확률['prob_1234'] = n_확률_1234
        df_번호5_확률['prob_15'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_15]
        df_번호5_확률['prob_25'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_25]
        df_번호5_확률['prob_35'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_35]
        df_번호5_확률['prob_45'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_45]
        ary_확률 = df_번호5_확률.loc[:, 'prob_1234': 'prob_45'].values
        df_번호5_확률['prob_total'] = [ary.prod() for ary in ary_확률]
        df_번호5_확률 = df_번호5_확률.sort_values('prob_total', ascending=False)

        n_번호5 = [int(s) for s in df_번호5_확률['no'].values[0].replace('no', '').split('|')
                 if int(s) is not n_번호4][0]
        li_번호조합.append(n_번호5)

        # 6번째 번호 찾기
        li_번호조합.sort()
        li_번호6 = [n for n in [n + 1 for n in range(45)]
                  if (int(n) is not n_번호1) and (int(n) is not n_번호2) and (int(n) is not n_번호3)
                  and (int(n) is not n_번호4) and (int(n) is not n_번호5)]

        n_확률_12345 = df_번호5_확률['prob_total'].values[0]
        li_번호조합_16 = [[n_번호1, n_번호6] if n_번호1 < n_번호6 else [n_번호6, n_번호1] for n_번호6 in li_번호6]
        li_번호조합_16 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_16]
        li_번호조합_26 = [[n_번호2, n_번호6] if n_번호2 < n_번호6 else [n_번호6, n_번호2] for n_번호6 in li_번호6]
        li_번호조합_26 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_26]
        li_번호조합_36 = [[n_번호3, n_번호6] if n_번호3 < n_번호6 else [n_번호6, n_번호3] for n_번호6 in li_번호6]
        li_번호조합_36 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_36]
        li_번호조합_46 = [[n_번호4, n_번호6] if n_번호4 < n_번호6 else [n_번호6, n_번호4] for n_번호6 in li_번호6]
        li_번호조합_46 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_46]
        li_번호조합_56 = [[n_번호5, n_번호6] if n_번호5 < n_번호6 else [n_번호6, n_번호5] for n_번호6 in li_번호6]
        li_번호조합_56 = [f'no{li_번호[0]:02}|{li_번호[1]:02}' for li_번호 in li_번호조합_56]

        df_번호6_확률 = pd.DataFrame()
        df_번호6_확률['no'] = li_번호조합_56
        df_번호6_확률['prob_12345'] = n_확률_12345
        df_번호6_확률['prob_16'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_16]
        df_번호6_확률['prob_26'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_26]
        df_번호6_확률['prob_36'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_36]
        df_번호6_확률['prob_46'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_46]
        df_번호6_확률['prob_56'] = [dic_확률_2개[s_번호조합] for s_번호조합 in li_번호조합_56]
        ary_확률 = df_번호6_확률.loc[:, 'prob_12345': 'prob_56'].values
        df_번호6_확률['prob_total'] = [ary.prod() for ary in ary_확률]
        df_번호6_확률 = df_번호6_확률.sort_values('prob_total', ascending=False)

        n_번호6 = [int(s) for s in df_번호6_확률['no'].values[0].replace('no', '').split('|')
                 if int(s) is not n_번호5][0]
        li_번호조합.append(n_번호6)

        # 번호조합 추가
        li_번호조합.sort()
        if li_번호조합 not in li_li_번호조합:
            li_li_번호조합.append(li_번호조합)

        # 번호조합 5개 완성 시 중지
        if len(li_li_번호조합) == 5:
            break

    # df 변환
    li_컬럼명 = [f'no{n + 1}' for n in range(6)]
    df_6개번호_5개세트 = pd.DataFrame(li_li_번호조합, columns=li_컬럼명)

    return s_전략명, df_6개번호_5개세트


# noinspection PyUnresolvedReferences,PyProtectedMember,PyPep8Naming
def 번호선정로직_복합로직_최빈수_따라가기(dic_정보):
    """ 최빈수 로직 + 따라가기 로직 합해서 번호 선정 """
    # function 이름 찾기
    func_name = sys._getframe(0).f_code.co_name
    s_전략명 = func_name

    # 확률값 가져오기
    df_확률_1개 = dic_정보['df_확률_1개']
    df_확률_2개 = dic_정보['df_확률_2개']
    df_확률_6개 = dic_정보['df_확률_6개']

    # 최빈수 로직으로 2개 번호 찾기
    # 2개 확률 중 상위 30개 번호 최빈수 찾기
    df_30개 = df_확률_2개[:30].copy()
    df_30개['no1'] = df_30개['no'].apply(lambda x: int(x[2:4]))
    df_30개['no2'] = df_30개['no'].apply(lambda x: int(x[5:7]))
    li_30개 = list(df_30개['no1'].values) + list(df_30개['no2'].values)
    sri_카운트 = pd.Series(li_30개).value_counts()
    n_최빈번호 = sri_카운트.index[0]

    # 2개 번호 골라내기
    ary_2개번호_30개 = df_30개.loc[:, ['no1', 'no2']].values
    li_2개번호_전체 = [list(ary_2개번호) for ary_2개번호 in ary_2개번호_30개 if n_최빈번호 in ary_2개번호]

    # 6개 번호조합에서 2개 번호 포함된 항목 골라내기
    ary_6개번호_전체 = df_확률_6개.loc[:, 'no1': 'no6'].values
    li_6개번호_전체 = [list(ary_6개번호) for ary_6개번호 in ary_6개번호_전체 if n_최빈번호 in ary_6개번호]
    li_2개번호_5개세트_최빈수 = []
    li_6개번호_5개세트_최빈수 = []
    for li_6개번호 in li_6개번호_전체:
        # 2개 번호 포함 여부 확인
        for li_2개번호 in li_2개번호_전체:
            if (li_2개번호[0] in li_6개번호) and (li_2개번호[1] in li_6개번호):
                li_2개번호_5개세트_최빈수.append(li_2개번호)
                li_6개번호_5개세트_최빈수.append(li_6개번호)
                li_2개번호_전체.remove(li_2개번호)
                break

        if len(li_6개번호_5개세트_최빈수) >= 5:
            break

    # 따라가기 로직으로 3개 번호 찾기
    # 예측번호 1개 확인
    ary_예측번호_1개_s = df_확률_1개[:5]['no'].values
    li_예측번호_1개_n = [int(s[2:]) for s in ary_예측번호_1개_s]

    # 예측번호 2개 숫자 분리
    df_확률_2개['no1'] = df_확률_2개['no'].apply(lambda x: int(x[2:4]))
    df_확률_2개['no2'] = df_확률_2개['no'].apply(lambda x: int(x[5:7]))

    # 1개확률 번호별 번호조합 생성
    li_li_번호조합 = []
    for n_번호1 in li_예측번호_1개_n:
        li_번호조합 = [n_번호1]
        # 2개번호 중 최고확률 찾기
        for i in range(5):
            n_번호_앞 = li_번호조합[-1]
            li_df_2개번호 = [df_확률_2개[df_확률_2개['no1'] == n_번호_앞], df_확률_2개[df_확률_2개['no2'] == n_번호_앞]]
            df_2개번호 = pd.concat(li_df_2개번호, axis=0).drop_duplicates().sort_values('prob_1', ascending=False)
            for j in range(5):
                ary_번호_뒤 = df_2개번호.loc[:, ['no1', 'no2']].values[j]
                n_번호_뒤 = [int(n) for n in ary_번호_뒤 if int(n) is not n_번호_앞][0]
                if n_번호_뒤 not in li_번호조합:
                    li_번호조합.append(n_번호_뒤)
                    break
                else:
                    continue
        li_li_번호조합.append(li_번호조합)

    # 최빈수 로직 + 따라가기 로직 합치기
    li_최빈수_2개 = li_2개번호_5개세트_최빈수
    li_최빈수_6개 = li_6개번호_5개세트_최빈수
    li_따라가기_6개 = li_li_번호조합

    # 2개 번호 첫번째 꺼로 번호 조합
    li_6개번호_5개세트_1 = []
    li_최빈수 = [int(s) for s in li_최빈수_2개[0]]
    for li_따라가기 in li_따라가기_6개:
        li_따라가기 = [n for n in li_따라가기 if n is not li_최빈수[0]]
        li_따라가기 = [n for n in li_따라가기 if n is not li_최빈수[1]]
        li_6개번호 = li_최빈수 + li_따라가기[:4]
        li_6개번호.sort()
        li_6개번호_5개세트_1.append(li_6개번호)

    # 2개 번호 두번째 꺼로 번호 조합
    li_6개번호_5개세트_2 = []
    li_최빈수 = [int(s) for s in li_최빈수_2개[1]]
    for li_따라가기 in li_따라가기_6개:
        li_따라가기 = [n for n in li_따라가기 if n is not li_최빈수[0]]
        li_따라가기 = [n for n in li_따라가기 if n is not li_최빈수[1]]
        li_6개번호 = li_최빈수 + li_따라가기[:4]
        li_6개번호.sort()
        li_6개번호_5개세트_2.append(li_6개번호)

    # 5개세트 생성
    li_6개번호_5개세트_12 = []
    for i in range(len(li_6개번호_5개세트_1 + li_6개번호_5개세트_2)):
        if i % 2 == 0:
            li_6개번호_5개세트_12.append(li_6개번호_5개세트_1[int(i / 2)])
        else:
            li_6개번호_5개세트_12.append(li_6개번호_5개세트_2[int(i / 2)])

    li_6개번호_5개세트 = []
    for li_6개번호 in li_6개번호_5개세트_12:
        if li_6개번호 not in li_6개번호_5개세트:
            li_6개번호_5개세트.append(li_6개번호)
        if len(li_6개번호_5개세트) >= 5:
            break

    # df 변환
    li_컬럼명 = [f'no{n + 1}' for n in range(6)]
    df_6개번호_5개세트 = pd.DataFrame(li_6개번호_5개세트, columns=li_컬럼명)

    return s_전략명, df_6개번호_5개세트


# noinspection PyUnresolvedReferences,PyProtectedMember,PyPep8Naming
def 번호선정로직_1개2개연계_최빈수연계(dic_정보):
    """ 1개확률 + 2개확률 번호 연계하여 최빈수 로직 적용하여 번호 선정 (6개확률 사용) """
    # function 이름 찾기
    func_name = sys._getframe(0).f_code.co_name
    s_전략명 = func_name

    # 확률값 가져오기
    df_확률_1개 = dic_정보['df_확률_1개']
    df_확률_2개 = dic_정보['df_확률_2개']
    df_확률_6개 = dic_정보['df_확률_6개']

    # no1 찾기 (1개 확률 top1)
    li_no = [int(s_no.replace('no', '')) for s_no in df_확률_1개['no'].values]
    n_no1 = li_no[0]
    li_no_제외 = li_no[1:6]

    # no2 찾기 (2개 확률 중 최빈수)
    df_30개 = df_확률_2개[:30].copy()
    df_30개['no1'] = df_30개['no'].apply(lambda x: int(x[2:4]))
    df_30개['no2'] = df_30개['no'].apply(lambda x: int(x[5:7]))
    li_30개 = list(df_30개['no1'].values) + list(df_30개['no2'].values)
    sri_카운트_2개확률 = pd.Series(li_30개).value_counts()
    n_no2 = int(sri_카운트_2개확률.index[0])
    if n_no2 == n_no1:
        n_no2 = int(sri_카운트_2개확률.index[1])

    ary_2개번호_30개 = df_30개.loc[:, ['no1', 'no2']].values
    li_2개번호_연계_페어 = [list(ary_2개번호) for ary_2개번호 in ary_2개번호_30개 if n_no2 in ary_2개번호]
    li_no_연계 = [[int(no) for no in li_2개번호 if int(no) is not n_no2][0] for li_2개번호 in li_2개번호_연계_페어]

    # no3, no4 찾기 (6개 확률 상위 1000개 중 no1, no2 동시에 포함 항목 => 연계 list 에 포함된 항목 중 빈도 top2)
    df_1000개 = df_확률_6개[:1000].copy()
    for n in range(6):
        df_1000개[f'no{n + 1}'] = df_1000개[f'no{n + 1}'].apply(lambda x: int(x))
    df_1000개['li_no'] = list(df_1000개.loc[:, 'no1': 'no6'].values)
    df_1000개['포함_no1'] = df_1000개['li_no'].apply(lambda li: (n_no1 in li) * 1)
    df_1000개['포함_no2'] = df_1000개['li_no'].apply(lambda li: (n_no2 in li) * 1)
    df_1000개['포함_동시'] = df_1000개['포함_no1'] + df_1000개['포함_no2']
    df_동시포함 = df_1000개[df_1000개['포함_동시'] == 2].copy()

    li_동시포함 = list(df_동시포함['no1'].values) + list(df_동시포함['no2'].values) + list(df_동시포함['no3'].values)\
              + list(df_동시포함['no4'].values) + list(df_동시포함['no5'].values) + list(df_동시포함['no6'].values)
    sri_카운트_6개확률 = pd.Series(li_동시포함).value_counts()
    li_연계포함 = [no for no in sri_카운트_6개확률.index if (no in li_no_연계) and (no not in [n_no1, n_no2])]
    n_no3 = li_연계포함[0] if len(li_연계포함) > 0 else 0
    n_no4 = li_연계포함[1] if len(li_연계포함) > 1 else 0

    # no5 찾기 (연계 list 중 미선택 no 선택(단, 제외 list 반영) => 미선택 no와 연계된 no list 추출)
    li_no1234 = [n_no1, n_no2, n_no3, n_no4]
    li_남은거 = [no for no in li_no_연계 if (no not in li_no1234) and (no not in li_no_제외)]

    li_no5 = []
    for n_남은거 in li_남은거:
        ary_2개번호_30개 = df_30개.loc[:, ['no1', 'no2']].values
        li_2개번호_연계_페어 = [list(ary_2개번호) for ary_2개번호 in ary_2개번호_30개 if n_남은거 in ary_2개번호]
        li_no_연계 = [[int(no) for no in li_2개번호 if int(no) is not n_남은거][0] for li_2개번호 in li_2개번호_연계_페어]
        for no in li_no_연계:
            li_no5.append(no)
    li_no5 = list(pd.Series(li_no5).drop_duplicates())
    li_no5 = [no for no in li_no5 if (no not in li_no1234) and (no not in li_no_제외)]

    # no6 찾기 (no3, no4의 연계 no list)
    ary_2개번호_30개 = df_30개.loc[:, ['no1', 'no2']].values

    li_2개번호_연계_페어 = [list(ary_2개번호) for ary_2개번호 in ary_2개번호_30개
                     if (n_no3 in ary_2개번호) or (n_no4 in ary_2개번호)]
    li_no_연계 = [[int(no) for no in li_2개번호 if int(no) is not n_no3] for li_2개번호 in li_2개번호_연계_페어]
    li_no_연계 = [[int(no) for no in li_2개번호 if int(no) is not n_no4] for li_2개번호 in li_no_연계]
    li_no_연계 = [li_no[0] for li_no in li_no_연계 if len(li_no) > 0]

    li_no6 = list(pd.Series(li_no_연계).drop_duplicates())
    li_no6 = [no for no in li_no6 if (no not in li_no1234) and (no not in li_no5) and (no not in li_no_제외)]

    # 번호 조합
    li_li_번호조합 = []
    li_번호조합_4개 = [n_no1, n_no2, n_no3, n_no4]
    if len(li_no5) == 0:
        li_li_번호조합.append(li_번호조합_4개 + [0, 0])
    for n_no5 in li_no5:
        li_번호조합_5개 = li_번호조합_4개 + [n_no5]
        if len(li_no6) == 0:
            li_li_번호조합.append(li_번호조합_5개 + [0])
        for n_no6 in li_no6:
            li_번호조합_6개 = li_번호조합_5개 + [n_no6]
            li_li_번호조합.append(li_번호조합_6개)

    li_cols = ['no1', 'no2', 'no3', 'no4', 'no5', 'no6']
    df_번호조합 = pd.DataFrame(li_li_번호조합, columns=li_cols)
    li_번호조합_정렬 = [list(pd.Series(li).sort_values()) for li in li_li_번호조합]
    df_번호조합_정렬 = pd.DataFrame(li_번호조합_정렬, columns=li_cols)

    df_6개번호_5개세트 = df_번호조합

    return s_전략명, df_6개번호_5개세트


# noinspection PyUnresolvedReferences,PyProtectedMember,PyPep8Naming
def 번호선정로직_앙상블_결과종합(dic_정보):
    """ 1개확률 + 2개확률 번호 연계하여 최빈수 로직 적용하여 번호 선정 (6개확률 사용) """
    # function 이름 찾기
    func_name = sys._getframe(0).f_code.co_name
    s_전략명 = func_name

    # 기준정보 가져오기
    folder_번호선정 = dic_정보['folder_번호선정']
    n_회차 = dic_정보['n_회차']
    s_추첨일 = dic_정보['s_추첨일']

    # 확률값 가져오기
    df_확률_1개 = dic_정보['df_확률_1개']
    df_확률_2개 = dic_정보['df_확률_2개']
    df_확률_6개 = dic_정보['df_확률_6개']

    # 선정 번호들 가져오기
    li_로직 = [폴더 for 폴더 in os.listdir(folder_번호선정) if '.csv' not in 폴더]
    li_로직 = [폴더 for 폴더 in li_로직 if s_전략명 not in 폴더]

    li_df_번호 = []
    for s_로직 in li_로직:
        s_폴더명 = os.path.join(folder_번호선정, s_로직)
        df_번호 = pd.read_csv(os.path.join(s_폴더명, f'확률예측_번호선정_{n_회차}차_{s_추첨일}추첨.csv'), encoding='cp949')
        li_df_번호.append(df_번호)

    df_번호 = pd.concat(li_df_번호, axis=0)

    # 중복 등장 찾기
    sri_카운트_6개 = df_번호.value_counts(ascending=False)
    sri_중복등장 = sri_카운트_6개[sri_카운트_6개 > 1].index
    li_중복등장 = [list(tup) for tup in sri_중복등장]

    # 최빈수 구하기
    sri_선정번호 = pd.Series(df_번호.values.reshape(1, -1)[0])
    sri_카운트_1개 = sri_선정번호.value_counts(ascending=False)
    li_상위4개 = list(sri_카운트_1개.index[:4])
    li_나머지 = list(sri_카운트_1개.index[4:])
    li_나머지 = [n for n in li_나머지 if n > 0]

    # 나머지 숫자로 2가지 조합하는 경우의 수 구하기
    li_li_2개 = []
    li_나머지_2번째 = [n for n in li_나머지]
    for n_번호1 in li_나머지:
        li_나머지_2번째.remove(n_번호1)
        dummy = [li_li_2개.append([n_번호1, n_번호2]) for n_번호2 in li_나머지_2번째]

    # 확률 확인할 숫자 조합 구하기
    li_6개번호 = [sorted(li_상위4개 + li_2개조합) for li_2개조합 in li_li_2개]
    df_6개번호 = pd.DataFrame(li_6개번호, columns=[f'no{n + 1}' for n in range(6)])

    # 확률 확인
    li_구분자 = ['|'] * len(df_확률_6개)
    df_확률_6개['key'] = df_확률_6개['no1'].astype(str) + li_구분자 + df_확률_6개['no2'].astype(str) + li_구분자\
                      + df_확률_6개['no3'].astype(str) + li_구분자 + df_확률_6개['no4'].astype(str) + li_구분자\
                      + df_확률_6개['no5'].astype(str) + li_구분자 + df_확률_6개['no6'].astype(str)
    dic_확률_6개 = df_확률_6개.set_index('key').to_dict()['prob_1']

    li_구분자 = ['|'] * len(df_6개번호)
    df_6개번호['key'] = df_6개번호['no1'].astype(str) + li_구분자 + df_6개번호['no2'].astype(str) + li_구분자\
                     + df_6개번호['no3'].astype(str) + li_구분자 + df_6개번호['no4'].astype(str) + li_구분자\
                     + df_6개번호['no5'].astype(str) + li_구분자 + df_6개번호['no6'].astype(str)
    df_6개번호['확률'] = df_6개번호['key'].apply(lambda x: dic_확률_6개[x] if x in dic_확률_6개.keys() else 0)
    df_6개번호 = df_6개번호.sort_values('확률', ascending=False)

    # 번호 선택 (중복 2개, 확률 하위 1개, 확률 상위 n개)
    ary_6개번호 = df_6개번호.loc[:, 'no1': 'no6'].values

    li_선정_중복 = [li_중복 + ['중복등장'] for li_중복 in li_중복등장[:2]]
    li_선정_하위 = [list(ary_6개번호[-1]) + ['확률하위']]

    n_상위갯수 = 5 - len(li_선정_중복 + li_선정_하위)
    li_선정_상위 = [list(ary) + ['확률상위'] for ary in ary_6개번호[: n_상위갯수]]
    li_선정번호 = li_선정_중복 + li_선정_상위 + li_선정_하위

    # df 정리
    li_cols = ['no1', 'no2', 'no3', 'no4', 'no5', 'no6', 'remark']
    df_6개번호_5개세트 = pd.DataFrame(li_선정번호, columns=li_cols)

    return s_전략명, df_6개번호_5개세트
