from pycallgraphix.wrapper import register_method
from pandas import DataFrame
from scipy.stats import shapiro, normaltest, bartlett, levene, ttest_1samp, ttest_ind, ttest_rel, mannwhitneyu, pearsonr, spearmanr
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from pingouin import anova, welch_anova, pairwise_tukey, pairwise_tests, pairwise_gameshowell
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pingouin import pairwise_tukey, pairwise_tests, pairwise_gameshowell
from .util import my_pretty_table, my_unmelt
from .plot import my_heatmap
from .core import *

@register_method
def my_normal_test(data: DataFrame, method: str = "n") -> None:
    """데이터프레임 내의 모든 컬럼에 대해 정규성 검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        method (str, optional): 정규성 검정 방법(n=normaltest, s=shapiro). Defaults to "n".
    """
    for c in data.columns:
        if method == "n":
            n = "normaltest"
            s, p = normaltest(data[c])
        else:
            n = "shapiro"
            s, p = shapiro(data[c])
            
        print(f"[{n}-{c}] statistic: {s:.3f}, p-value: {p:.3f}, 정규성 충족 여부: {p > 0.05}")

@register_method
def my_equal_var_test(data: DataFrame, normal_dist: bool = True) -> None:
    """데이터프레임 내에 있는 모든 컬럼들에 대해 등분산성 검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        normal_dist (bool, optional): 정규성 검정 결과를 의미한다. True일 경우 정규분포를 따르는 데이터에 대한 등분산성 검정을 수행한다. Defaults to True.
    """
    fields: list = [data[x] for x in set(data.colunms)]

    if normal_dist:
        n: str = "Bartlett"
        s, p = bartlett(*fields)
    else:
        n: str = "Levene"
        s, p = levene(*fields)
        
    print(f"{n} 검정: statistic: {s:.3f}, p-value: {p:.3f}, 등분산성 충족 여부: {p > 0.05}")

@register_method
def my_normal_equal_var_1field(data: DataFrame, xname: str = 'x', hue: str = 'hue') -> None:
    """데이터프레임 내에 있는 한 종류의 명목형 변수에 따라 종속변수의 정규성과 등분산성을 검정하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str, optional): 종속변수의 컬럼명. Defaults to 'x'.
        hue (str, optional): 명목형 변수의 컬럼명. Defaults to 'hue'.
    """
    u1 = data[hue].unique()
    equal_var_fields: list = []
    normal_dist: bool = True
    report: list = []

    for i in u1:
        filtered_data = data[data[hue] == i][xname]
        equal_var_fields.append(filtered_data)
        s, p = normaltest(filtered_data)

        normalize = p > 0.05
        report.append({"field": i, "statistic": s, "p-value": p, "result": normalize})
        normal_dist = normal_dist and normalize

    if normal_dist:
        n = "Bartlett"
        s, p = bartlett(*equal_var_fields)
    else:
        n = "Levene"
        s, p = levene(*equal_var_fields)
        
    report.append({"field": n, "statistic": s, "p-value": p, "result": p > 0.05})
    report_df = DataFrame(report).set_index('field')
    my_pretty_table(report_df)
    
@register_method
def my_normal_equal_var_2field(data: DataFrame, xname: str = 'x', hue: list = ['h1', 'h2']) -> None:
    """데이터프레임 내에 있는 두 종류의 명목형 변수에 따라 종속변수의 정규성과 등분산성을 검정하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str, optional): 종속변수의 컬럼명. Defaults to 'x'.
        hue (list, optional): 명목형 변수의 컬럼명을 저장하고 있는 리스트. Defaults to ['h1', 'h2'].
    """
    u1 = data[hue[0]].unique()
    u2 = data[hue[1]].unique()
    equal_var_fields: list = []
    normal_dist: bool = True
    report: list = []

    for i in u1:
        for j in u2:
            filtered_data = data[(data[hue[0]] == i) & (data[hue[1]] == j)][xname]
            equal_var_fields.append(filtered_data)
            s, p = normaltest(filtered_data)

            normalize: bool = p > 0.05
            report.append({"field": f"{i}, {j}", "statistic": s, "p-value": p, "result": normalize})
            normal_dist = normal_dist and normalize

    if normal_dist:
        n: str = "Bartlett"
        s, p = bartlett(*equal_var_fields)
    else:
        n: str = "Levene"
        s, p = levene(*equal_var_fields)
        
    report.append({"field": n, "statistic": s, "p-value": p, "result": p > 0.05})
    report_df = DataFrame(report).set_index('field')
    my_pretty_table(report_df)

@register_method
def my_ttest_1samp(data: DataFrame, mean_value: int = 0) -> None:
    """데이터프레임 내에 있는 모든 컬럼에 대해 일표본 t-검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        mean_value (int, optional): 귀무가설의 기준값. Defaults to 0.
    """
    alternative: list = ["two-sided", "less", "greater"]
    result: list = []

    for c in data.columns:
        for a in alternative:
            s, p = ttest_1samp(data[c], mean_value, alternative=a)
            
            itp = None
            
            if a == "two-sided":
                itp = f"μ {'==' if p > 0.05 else '!='} {mean_value}"
            elif a == "less":
                itp = f"μ {'>=' if p > 0.05 else '<'} {mean_value}"
            else:
                itp = f"μ {'<=' if p > 0.05 else '>'} {mean_value}"
            
            result.append({
                "field": c,
                "alternative": a,
                "statistic": round(s, 3),
                "p-value": round(p, 3),
                "H0": p > 0.05,
                "H1": p <= 0.05,
                "interpretation": itp
            })
            
    rdf = DataFrame(result).set_index(["field", "alternative"])
    my_pretty_table(rdf)

@register_method
def my_ttest_ind(data: DataFrame, xname: str, yname: str, equal_var: bool = True) -> None:
    """독립표본 t-검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str): 첫 번째 필드 이름
        yname (str): 두 번째 필드 이름
        equal_var (bool, optional): 등분산성 가정 여부. Defaults to True.
    """
    
    alternative: list = ["two-sided", "less", "greater"]
    result: list = []
    fmt: str = "μ({f0}) {0} μ({f1})"
    
    for a in alternative:
        s, p = ttest_ind(data[xname], data[yname], equal_var=equal_var, alternative=a)
        n = "t-test_ind" if equal_var else "Welch's t-test"
            
        # 검정 결과 해석
        itp = None
        
        if a == "two-sided":
            itp = fmt.format("==" if p > 0.05 else "!=", f0=xname, f1=yname)
        elif a == "less":
            itp = fmt.format(">=" if p > 0.05 else "<", f0=xname, f1=yname)
        else:
            itp = fmt.format("<=" if p > 0.05 else ">", f0=xname, f1=yname)
            
        result.append({
            "test": n,
            "alternative": a,
            "statistic": round(s, 3),
            "p-value": round(p, 3),
            "H0": p > 0.05,
            "H1": p <= 0.05,
            "interpretation": itp
        })
        
    rdf = DataFrame(result).set_index(["test", "alternative"])
    my_pretty_table(rdf)

@register_method
def my_ttest_rel(data: DataFrame, xname: str, yname: str, equal_var: bool = True) -> None:
    """대응표본 t-검정 또는 Mann-Whitney U 검정을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str): 첫 번째 필드 이름
        yname (str): 두 번째 필드 이름
        equal_var (bool, optional): 등분산성 가정 여부. True인 경우 대응표본 T검정 수행 / False인 경우 Mann-Whitney 검정 수행. Defaults to True.
    """
    
    alternative: list = ["two-sided", "less", "greater"]
    result: list = []
    fmt: str = "μ({f0}) {0} μ({f1})"
    
    for a in alternative:
        if equal_var:
            s, p = ttest_rel(data[xname], data[yname], alternative=a)
            n = "t-test_paired"
        else:
            s, p = mannwhitneyu(data[xname], data[yname], alternative=a)
            n = "Mann-Whitney"
            
        # 검정 결과 해석
        itp = None
        
        if a == "two-sided":
            itp = fmt.format("==" if p > 0.05 else "!=", f0=xname, f1=yname)
        elif a == "less":
            itp = fmt.format(">=" if p > 0.05 else "<", f0=xname, f1=yname)
        else:
            itp = fmt.format("<=" if p > 0.05 else ">", f0=xname, f1=yname)
            
        result.append({
            "test": n,
            "alternative": a,
            "statistic": round(s, 3),
            "p-value": round(p, 3),
            "H0": p > 0.05,
            "H1": p <= 0.05,
            "interpretation": itp
        })
        
    rdf = DataFrame(result).set_index(["test", "alternative"])
    my_pretty_table(rdf)
    
@register_method
def my_anova(data: DataFrame, target: str, hue: any, equal_var: bool = True, post : bool = False) -> None:
    """분산분석을 수행하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        target (str): 종속변수의 컬럼명
        hue (_type_): 명목형 변수의 컬럼명을 저장하고 있는 리스트
        equal_var (bool, optional): 등분산성 가정 여부. Defaults to True.
        post (bool, optional): 사후검정 여부, Defaults to False
    """
    
    # 일원 분산 분석 명목형 파라미터 정리
    if type(hue) == str or type(hue) == list and len(hue) == 1:
        anova_type = "oneway"
        expr = f'{target} ~ C({hue})'
        typ = 1
        if (type(hue) == list):
            hue = hue[0]
    # 이원 분산 분석 -> type(hue) == list and len(hue) > 1:
    else:
        anova_type = "twoway"
        expr = f'{target} ~ '
        for i, h in enumerate(hue):
            expr += f'C({h})'
            if i + 1 < len(hue):
                expr += '*'
        typ = 2
    # pingouin 패키지를 사용하는 경우
    if equal_var:
        # 등분산성을 충족하는 경우에 대해서는 oneway, twoway 분석 모두 지원
        print("pingouin.anova")
        aov = anova(data=data, dv=target, between=hue, detailed=True)
        my_pretty_table(aov)
    else:
        # 등분산성을 충족하지 않는 경우에 대해서는 oneway 분석만 지원
        if anova_type == "oneway":
            print("pingouin.welch_anova")
            aov = welch_anova(data=data, dv=target, between=hue)
            my_pretty_table(aov)
    
    # statsmodels 패키지를 사용하는 경우
    print("\nstatsmodels.anova.anova_lm")  
    lm = ols(expr, data=data).fit()
    if equal_var:
        anova_result = anova_lm(lm, typ=typ)
    else:
        anova_result = anova_lm(lm, typ=typ, robust="hc3")
    my_pretty_table(anova_result)
    
    s = anova_result['F'][0]
    p = anova_result['PR(>F)'][0]
    print(f"[anova_lm] statistic: {s:.3f}, p-value: {p:.3f}, {'대립' if p <= 0.05 else '귀무'}가설 채택")
    
    # 일원 분산 분석인 경우 사후검정 수행
    if post and anova_type == "oneway":
        # 등분산인 경우
        if equal_var:
            cnt = data[[target, hue]].groupby(hue).count()
            
            # 샘플수가 같은 경우 투키 방법
            if cnt.iloc[0, 0] == cnt[cnt.columns[0]].mean():
                print("\n사후검정: Tukey HSD 방법")
                mc = MultiComparison(data[target], data[hue])
                result = mc.tukeyhsd()
                print(result)
            # 샘플수가 다른 경우 본페로니 방법
            else:
                print("\n사후검정: 본페로니 방법")
                result = pairwise_tests(data=data, dv=target, between=hue, padjust='bonf')
                my_pretty_table(result)
        
        # 등분산이 아닌 경우 -> gameshowell
        else:
            print("\n사후검정: Games-Howell 방법")
            result = pairwise_gameshowell(data=data, dv=target, between=hue)
            my_pretty_table(result)
   
@register_method
def my_correlation(data: DataFrame, method: str = "p", heatmap: bool = True, figsize: list=(10, 8), dpi: int=150) -> None:
    """데이터프레임 내에 있는 모든 컬럼들에 대해 상관계수를 계산하고 결과를 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        method (str, optional): 상관계수 계산 방법(p=pearson, s=spearman). Defaults to "p".
        heatmap (bool, optional): 상관계수 히트맵 출력 여부. Defaults to True.
        figsize (list, optional): 히트맵의 크기. Defaults to (10, 8).
        dpi (int, optional): 히트맵의 해상도. Defaults to 150.
    """
    if heatmap:
        my_heatmap(data.corr(method="pearson" if method == "p" else "spearman"), figsize=figsize, dpi=dpi)
    else:
        my_pretty_table(data.corr(method="pearson" if method == "p" else "spearman"))
    
    result = []
    
    for i,c in enumerate(data.columns):
        for j,d in enumerate(data.columns):
            if i < j :
                if method == 'p':
                    s, p = pearsonr(data[c], data[d])
                else:
                    s, p = spearmanr(data[c], data[d])
                    
                result.append({
                    "field1": c,
                    "field2": d,
                    "correlation": s,
                    "p-value": p,
                    "result": p <= 0.05
                })
                
    rdf = DataFrame(result)
    rdf.set_index(["field1", "field2"], inplace=True)
    my_pretty_table(rdf)

