import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#sns.set_theme(style="darkgrid")
sns.set(font_scale = 2)
sns.set_style("whitegrid", {'axes.grid' : True})

def plot_align(df, lang):
    # choose the highest score for each size, categorized by alignment type
    ret = []
    for alignment_method in ["hunalign", "laser", "sbert"]:
        alignment_out = []
        align_df = df[df['Align']==alignment_method]
        for size in [2,3,5,7]:
            align_df_size = align_df[align_df['Size']==size]
            highest_score = np.max(align_df_size['Score'])
            alignment_out.append(highest_score)
            # print(align_df_size.head())
            # print(highest_score)
        ret.append(alignment_out)
    print(ret)
    if lang == "ps":
        ret.append([9.2,9.4,9.7,8.8])
    else:
        ret.append([9.8,10.6,11,10.5])

    # plot_df =  pd.DataFrame(ret, columns=[2,3,5,7], index=["hunalign", "laser", "sbert"])
    plt.rcParams["figure.figsize"] = [10.00, 10.00]
    plt.rcParams["figure.autolayout"] = True
    plt.xlabel("EN Token Size (Million)")
    if lang == "ps":
        plt.ylabel("BLEU SCORE")
    plt.title(f"Alignment Comparison ({lang})")
    sns.lineplot(x=[2,3,5,7], y=ret[0], linestyle="dashed", marker="o", label="Hunalign")
    sns.lineplot(x=[2, 3, 5, 7], y=ret[1], linestyle="dashed", marker="o", label="LASER Align")
    sns.lineplot(x=[2, 3, 5, 7], y=ret[2], linestyle="dashed", marker="s", label="SBERT Align")
    sns.lineplot(x=[2, 3, 5, 7], y=ret[3], linestyle="dashed", marker="o", label="Best Score from WMT20")
    # sns.lineplot(x=[2,3,5,7], y=ret[0], linestyle="dashed", marker="o")
    # sns.lineplot(x=[2, 3, 5, 7], y=ret[1], linestyle="dashed", marker="o")
    # sns.lineplot(x=[2, 3, 5, 7], y=ret[2], linestyle="dashed", marker="s")
    # sns.lineplot(x=[2, 3, 5, 7], y=ret[3], linestyle="dashed", marker="o")
    if lang == "km":
        plt.legend()
    #plt.show()
    plt.savefig(f'align-{lang}.pdf', dpi=1000)

def plot_filter(df):
    scores_ps = []
    scores_km = []
    ps_df = df[df['Lang']=='ps']
    laser_align_df = ps_df[ps_df['Align']=="laser"]
    laser_align_laser_filter = laser_align_df[laser_align_df['Filter']=="laser"]
    scores_ps.append(laser_align_laser_filter['Score'].to_numpy().tolist())
    sbert_align_df = ps_df[ps_df['Align'] == "sbert"]
    sbert_align_sbert_filter = sbert_align_df[sbert_align_df['Filter'] == "sbert"]
    scores_ps.append(sbert_align_sbert_filter['Score'].to_numpy().tolist())

    km_df = df[df['Lang'] == 'km']
    laser_align_df = km_df[km_df['Align'] == "laser"]
    laser_align_laser_filter = laser_align_df[laser_align_df['Filter'] == "laser"]
    scores_km.append(laser_align_laser_filter['Score'].to_numpy().tolist())
    sbert_align_df = km_df[km_df['Align'] == "sbert"]
    sbert_align_sbert_filter = sbert_align_df[sbert_align_df['Filter'] == "sbert"]
    scores_km.append(sbert_align_sbert_filter['Score'].to_numpy().tolist())

    plt.rcParams["figure.figsize"] = [10.00, 10.00]
    plt.rcParams["figure.autolayout"] = True
    plt.xlabel("EN Token Size (Million)")
    plt.ylabel("BLEU SCORE")
    plt.title("LASER vs SBERT")
    sns.lineplot(x=[2, 3, 5, 7], y=scores_ps[0], linestyle="dashed", marker="o", label="LASER Align LASER Filter (ps)")
    sns.lineplot(x=[2, 3, 5, 7], y=scores_km[0], linestyle="dashed", marker="o", label="LASER Align LASER Filter (km)")
    sns.lineplot(x=[2, 3, 5, 7], y=scores_ps[1], linestyle="dashed", marker="s", label="SBERT Align SBERT Filter (ps)")
    sns.lineplot(x=[2, 3, 5, 7], y=scores_km[1], linestyle="dashed", marker="s", label="SBERT Align SBERT Filter (km)")
    plt.legend()
    #plt.show()
    plt.savefig(f'laser-sbert.pdf', dpi=1000)


def main():
    # visualize line plot for different alignment method
    df = pd.read_csv("experiment_result.csv")
    df = df[df['Type']=='test']
    ps_all_data = df[df['Lang']=='ps']
    km_all_data = df[df['Lang']=='km']
    #plot_align(ps_all_data, "ps")
    plot_align(km_all_data, "km")
    #plot_filter(df)


    #plot_filter()
    #print(df.head())

if __name__ == '__main__':
    main()