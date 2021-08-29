import numpy as np
import pandas as pd
import os.path as osp
import multiprocessing as mp
import os, datetime, matplotlib, holidays, shutil, time, csv, traceback, pickle, math, folium
from datetime import datetime
from itertools import product
from folium.features import DivIcon
from folium.map import Marker
import matplotlib.pyplot as plt
from utils import filter_nondigit, haversine_np

def validate_lent_data_by_year(df, tgt_year):
    mask1 = ((df['반납일시'] < str(tgt_year)) | (df['반납일시'] > str(tgt_year + 1)))
    mask2 = ((df['대여일시'] < str(tgt_year)) | (df['대여일시'] > str(tgt_year + 1)))
    print("반납일시, 대여일시가 2020년으로 되어있지 않은 row - 반납일시 {} | 대여일시 {}".format(mask1.sum(), mask2.sum()))

def split_by_max_row(data_path, filename):
    ## except header row, 1048575 rows are maximum rows in excel
    ## split into maximum rows
    MAX_ROW = 1048575
    df = read_csv(osp.join(data_path, filename))
    num_splits = (len(df) - 1) // MAX_ROW + 1
    for idx in range(num_splits):
        new_filename = filename[:-4]+'_{}.tsv'.format(idx+1)
        print(new_filename)
        start_idx, end_idx = idx * MAX_ROW, (idx+1) * MAX_ROW
        df.iloc[start_idx:end_idx].to_csv(osp.join(data_path, new_filename), encoding='CP949', index=False, sep='\t')

def read_csv(filepath, sep=','):
    if not osp.isfile(filepath):
        raise Exception("{} 존재하지 않습니다.".format(filepath))
    print("{} 파일을 로딩합니다.".format(filepath))
    # return pd.read_csv(filepath, engine='python',encoding='CP949')
    # return pd.read_csv(filepath, sep='\",\"', engine='python', encoding='CP949')
    return pd.read_csv(filepath, sep=sep, engine='python', encoding='CP949')

def read_station_loc_csv(station_loc_data_path, station_loc_filename='공공자전거 대여소 정보(21.06월 기준).csv'):
    station_loc_raw_df = read_csv(osp.join(station_loc_data_path, station_loc_filename))
    station_loc_raw_df = preprocess_station_loc_table(station_loc_raw_df)
    return station_loc_raw_df

def preprocess_lent_amount_table(df):
    df['대여일시'] = pd.to_datetime(df['대여일시'])
    df['반납일시'] = pd.to_datetime(df['반납일시'])
    return df
    
def preprocess_station_loc_table(ori_table):
    station_loc_df = ori_table.copy()
    ## 컬럼 이름 바꾸기
    new_column = ['대여소 번호', '보관소(대여소)명', '소재지(자치구)', '상세주소', 
              '위도', '경도', '설치 시기', 'LCD 거치대수', 'QR 거치대수', '운영 방식']
    station_loc_df.columns = new_column
    station_loc_df = station_loc_df.drop([0, 1, 2, 3], axis='index').reset_index(drop=True)

    ## 거치 대수 컬럼 합치기
    lcd_cnt = station_loc_df['LCD 거치대수'].fillna(0).to_numpy().astype(int)
    qr_cnt = station_loc_df['QR 거치대수'].fillna(0).to_numpy().astype(int)
    print('LCD 와 QR 거치대 모두 없는 대여소 개수: {}'.format(np.logical_and((lcd_cnt == 0), (qr_cnt == 0)).sum()))
    new_cnt = lcd_cnt + qr_cnt
    station_loc_df = station_loc_df.drop(['LCD 거치대수', 'QR 거치대수'], axis='columns')
    station_loc_df['거치대수'] = new_cnt
    
    ## 대여소 번호 데이터 타입 float -> int 로 변환 
    station_loc_df['대여소 번호'] = station_loc_df['대여소 번호'].astype(int)    
    
    ## 위도, 경도 데이터 타입 str -> float 로 변환
    station_loc_df['위도'] = station_loc_df['위도'].astype(float)    
    station_loc_df['경도'] = station_loc_df['경도'].astype(float)    
    
    ## 설치 시기 데이터 타입 str -> datetime 로 변환
    station_loc_df['설치 시기'] = pd.to_datetime(station_loc_df['설치 시기'])
#     station_loc_df['설치 시기'] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d'), station_loc_df['설치 시기']))

    print("자전거 대여소 위치 정보 데이터 전처리 완료")
    return station_loc_df

def merge_lent_amount_with_location_table(station_loc_df, lent_amount_df):
    # lend_station_loc = station_loc_df[['대여소 번호', '위도', '경도', '보관소(대여소)명', '소재지(자치구)']].set_axis(
    #     ['대여 대여소번호', '대여 대여소 위도', '대여 대여소 경도', '대여소 이름' , '자치구'], axis=1)
    lend_station_loc = station_loc_df[['대여소 번호', '위도', '경도']].set_axis(['대여 대여소번호', '대여 대여소 위도', '대여 대여소 경도'], axis=1)
    return_station_loc = station_loc_df[['대여소 번호', '위도', '경도']].set_axis(['반납대여소번호', '반납 대여소 위도', '반납 대여소 경도'], axis=1)
    lent_amount_df_with_loc = pd.merge(lent_amount_df, lend_station_loc, on='대여 대여소번호', how='inner')
    lent_amount_df_with_loc = pd.merge(lent_amount_df_with_loc, return_station_loc, on='반납대여소번호', how='inner')
    print("매칭된 대여소 위치 정보가 없어서 삭제된 대여 기록량: {}".format(len(lent_amount_df) - len(lent_amount_df_with_loc)))
    return lent_amount_df_with_loc


def bicycle_ids(df):
    return df['자전거번호'].unique()

def filter_corrupted_station_loc_table(df):
    corrupted_row_mask = (df['위도'] == 0.0) | (df['경도'] == 0.0)
    corrupted_rows = df[corrupted_row_mask]
    print("Drop corrupted data rows - #rows {}".format(corrupted_row_mask.sum()))
    df = df.drop(df.index[corrupted_row_mask]).reset_index(drop=True)
    return corrupted_rows, df

def find_corrupted_rows_idx_lent_2020_data(new_filelines):
    equal_len_mask = [len(l) == len(new_filelines[0]) for l in new_filelines] 
    equal_len_mask_np = np.array(equal_len_mask)
    corrupted_rows_idx = np.where(~equal_len_mask_np)[0]
    print("# corrupted row: {} - idx: {}".format(len(corrupted_rows_idx), corrupted_rows_idx))
    return corrupted_rows_idx

def modify_corrupted_rows_lent_2020_data(new_filelines, corrupted_rows_idx, station_loc_df):
    failed_idx = []
    for i, corr_idx in enumerate(corrupted_rows_idx):
        r = new_filelines[corr_idx]
        if r[6].isdigit(): ## 뒤에 정류소가 잘못된 경우
            tgt_idx = 6
        elif r[7].isdigit(): ## 앞에 정류소가 잘못된 경우
            tgt_idx = 2
        station_mask = station_loc_df['대여소 번호'] == int(r[tgt_idx])
        if station_mask.sum() == 0: ## 해당하는 대여소가 없는 경우
            failed_idx.append(corr_idx)
            print("[Fail to correct data - due to unmatched station] {} | id = {}".format(r[tgt_idx+1], r[tgt_idx]))
        else:
            num_stands = filter_nondigit(r[tgt_idx+1].split(',')[-1])
            correct_name = station_loc_df[station_mask]['보관소(대여소)명'].values[0]
            r[tgt_idx+1] = correct_name
            r.insert(tgt_idx+2, num_stands)
            print("[Corrected data] {} | {}".format(num_stands, correct_name))

    ## update target idx list if removed row is in front of the targeted row
    num_failed_idx = len(failed_idx)
    if num_failed_idx > 0:
        for i in range(num_failed_idx):
            new_filelines.pop(failed_idx[i])
            for j in range(i+1, num_failed_idx):
                if failed_idx[i] < failed_idx[j]:
                    failed_idx[j] -= 1

def convert_csv2tsv_corrupted_lent_2020_data(data_path, filenames, station_loc_df):
    failed_filenames = []
    for tgt_filename in filenames:
        try:
            print("Correcting {}".format(tgt_filename))
            start_time = time.time()
            new_filelines = []
            with open(osp.join(data_path, tgt_filename), 'r', errors='ignore') as fp:
                df_new = csv.reader(fp, delimiter=',')
                for idx, row in enumerate(df_new):
                    new_filelines.append(row)
            print('{} row 개수: {}'.format(tgt_filename, len(new_filelines)))
            corrupted_rows_idx = find_corrupted_rows_idx_lent_2020_data(new_filelines)
            if len(corrupted_rows_idx) > 0 :
                modify_corrupted_rows_lent_2020_data(new_filelines, corrupted_rows_idx, station_loc_df)
                new_filename = tgt_filename[:-4] + '_fixed' + '.tsv'
                with open(osp.join(data_path, new_filename), 'w') as fp:
                    fp.writelines(["%s\n"%('\t'.join(row)) for row in new_filelines])
            end_time = time.time()
            print("{} - {:.1f} 초 걸림".format(tgt_filename, end_time - start_time))
            df = read_csv(osp.join(data_path, new_filename), sep='\t')
            validate_lent_data_by_year(df, 2020)
        except Exception as e:
            print("Failed to correct {}".format(tgt_filename))
            print(e)
            print(''.join(traceback.format_tb(e.__traceback__)))
            failed_filenames.append(tgt_filename)

def count_daily_usage_of_month(df, year, month):
    tgt_days, daily_usage_cnts, is_holidays, is_weekends = [], [], [], []
    df_days = np.array(list(map(lambda d: d.day, df['반납일시'])))
    recorded_days = np.sort(np.unique(df_days))
    print("{}년 {:02d}월 데이터에 기록된 day: {}".format(year, month, list(recorded_days)))
    for tgt_day in recorded_days:
        tgt_date = datetime(year, month, tgt_day)
        is_holiday = tgt_date in holidays.KR()
        is_weekend = not is_holiday and tgt_date.weekday() > 4
        tgt_df = df[df_days == tgt_day]
        tgt_days.append(tgt_day)
        daily_usage_cnts.append(len(tgt_df))
        is_holidays.append(is_holiday)
        is_weekends.append(is_weekend)
        
    is_holidays = np.array(is_holidays)
    is_weekends = np.array(is_weekends)
    daily_usage_cnts = np.array(daily_usage_cnts)
    return daily_usage_cnts, tgt_days, is_holidays, is_weekends

def count_monthly_mean(daily_usage_cnts, is_holidays, is_weekends):
    monthly_mean_dict = {'일일': daily_usage_cnts.mean(),
                         '평일':daily_usage_cnts[~np.logical_or(is_holidays, is_weekends)].mean(),
                         '주말':daily_usage_cnts[is_weekends].mean(),
                         '공휴일':daily_usage_cnts[is_holidays].mean() if is_holidays.sum() > 0 else 0}
    return monthly_mean_dict

def print_monthly_mean(year, month, monthly_mean_dict):
    print("{}-{:02d} 평균 일일 대여량 - {:.2f}".format(year, month, monthly_mean_dict['일일']))
    print("{}-{:02d} 평균 평일 대여량 - {:.2f}".format(year, month, monthly_mean_dict['평일']))
    print("{}-{:02d} 평균 주말 대여량 - {:.2f}".format(year, month, monthly_mean_dict['주말']))
    print("{}-{:02d} 평균 공휴일 대여량 - {:.2f}".format(year, month, monthly_mean_dict['공휴일']))

def get_table_monthly_mean_usage_month(tgt_table, year, month):
    years, months, day_types, monthly_mean_values, day_cnts = [], [], [], [], []
    daily_usage_cnts, tgt_days, is_holidays, is_weekends = count_daily_usage_of_month(tgt_table, int(year), month)
    monthly_mean_dict = count_monthly_mean(daily_usage_cnts, is_holidays, is_weekends)
    all_day_cnt, weekend_cnt, holiday_cnt = len(tgt_days), is_weekends.sum(), is_holidays.sum()
    weekday_cnt = all_day_cnt - weekend_cnt - holiday_cnt
    day_cnt_dict = {'일일': all_day_cnt, '평일': weekday_cnt, '주말': weekend_cnt, '공휴일': holiday_cnt}
    for day_type in monthly_mean_dict.keys():
        years.append(year)
        months.append(month)
        day_types.append(day_type)
        monthly_mean_values.append(monthly_mean_dict[day_type])
        day_cnts.append(day_cnt_dict[day_type])
    monthly_mean_df = pd.DataFrame({
        'year': years,
        'month': months,
        'day_type': day_types,
        'monthly_mean': monthly_mean_values,
        'day_cnt': day_cnts
    })
    return monthly_mean_df

def get_table_monthly_mean_usage_all_month(dfs_year_group):
    output_dfs = []
    st_time_total = time.time()
    for year in dfs_year_group.keys():
        year_list = [int(year)] * len(dfs_year_group[year])
        month_list = list(range(1, len(dfs_year_group[year])+1))
        st_time = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            output_df = pool.starmap(get_table_monthly_mean_usage_month, zip(dfs_year_group[year], year_list, month_list))
        output_dfs.extend(output_df)
        print("get_table_monthly_mean_usage {}year takes {:.1f} sec".format(year, time.time() - st_time))
    monthly_mean_df = pd.concat(output_dfs)
    monthly_mean_df['monthly_sum'] = monthly_mean_df['monthly_mean'] * monthly_mean_df['day_cnt']
    print("get_table_monthly_mean_usage_all_month takes {:.1f} sec".format(time.time() - st_time_total))
    return monthly_mean_df

def plot_daily_rental_volume_graph(daily_usage_cnts, tgt_days, is_holidays, is_weekends, year, month):
    colors, labels = [], []
    for i in range(len(tgt_days)):
        tgt_color = 'black'
        if is_holidays[i]:
            tgt_color = 'blue'
        elif is_weekends[i]:
            tgt_color = 'red'
        colors.append(tgt_color)
        
    x = np.arange(len(tgt_days))
    plt.bar(x, daily_usage_cnts, color=colors)
    plt.xticks(x, tgt_days)
    plt.xlabel("일")
    plt.ylabel("대여량")
    plt.title('{}-{:02d} 일일 자전거 대여량'.format(year, month))
    colors = {'weekday':'black', 'holiday':'blue', 'weekend':'red'}         
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in colors.keys()]
    plt.legend(handles, list(colors.keys()))
    plt.show()

def plot_daily_rental_volume_graph_with_given_year_group(dfs_year_group):
    for year in dfs_year_group.keys():
        for month in range(len(dfs_year_group[year])):
            tgt_table = dfs_year_group[year][month]
            daily_usage_cnts, tgt_days, is_holidays, is_weekends = count_daily_usage_of_month(tgt_table, int(year), month + 1)
            plot_daily_rental_volume_graph(daily_usage_cnts, tgt_days, is_holidays, is_weekends, int(year), month + 1)

def find_tsv_filenames_by_year_group(data_path):
    filenames_year_group = {}
    for filename in [s for s in os.listdir(data_path) if s.endswith('.tsv')]:
        print(filename)
        year, month = filename.split('_')[1].split('.')[0:2]
        if year not in filenames_year_group:
            filenames_year_group[year] = []
        filenames_year_group[year].append(filename)
    for year in filenames_year_group.keys():
        filenames_year_group[year].sort()
    return filenames_year_group

def preprocess_and_load_tsv_lent_file(data_path, filename):
    st_time = time.time()
    df = preprocess_lent_amount_table(read_csv(osp.join(data_path, filename), sep='\t'))
    print('Load {} takes {:.1f} sec'.format(filename, time.time() - st_time))
    return df

def load_year_grouped_tsv_lent_files(filenames_year_group, data_path):
    dfs_year_group = {}
    st_time_total = time.time()
    for year in filenames_year_group.keys():
        filenames = filenames_year_group[year]
        st_time = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            dfs = pool.starmap(preprocess_and_load_tsv_lent_file, zip([data_path]*len(filenames), filenames))
            print("Loading year {} takes {:.1f} sec".format(year, time.time() - st_time))
        dfs_year_group[year] = dfs
    print("Loading all tsv lent files takes {:.1f} sec".format(time.time() - st_time_total))
    return dfs_year_group

def count_monthly_day_types(df, year, month):
    st_time = time.time()
    tgt_date_str = '{}-{}'.format(year, month+1)
    kr_holiday = holidays.KR()
    weekday_cnt, weekend_cnt, holiday_cnt = 0, 0, 0
    return_date_str_list = df['반납일시'].apply(lambda x: '{}-{}-{}'.format(x.year, x.month, x.day)).unique()
    for return_date_str in return_date_str_list:
        if tgt_date_str in return_date_str:
            return_date_datetime = datetime.strptime(return_date_str, '%Y-%m-%d')
            if return_date_datetime in kr_holiday:
                holiday_cnt += 1
            elif return_date_datetime.weekday() > 4:
                weekend_cnt += 1
            else:
                weekday_cnt += 1
    day_type_cnt = {'주말': weekend_cnt, '평일': weekday_cnt, '공휴일': holiday_cnt, '일일': weekend_cnt + weekday_cnt + holiday_cnt}
    print("Counting {}-{} day type takes {:.1f} sec".format(year, month+1, time.time() - st_time))
    return day_type_cnt

def count_monthly_day_types_cnt_year_group(dfs_year_group):
    monthly_day_types_cnt_year_group = {}
    for year in dfs_year_group.keys():
        num_months = len(dfs_year_group[year])
        year_list = [year] * num_months
        month_list = list(range(len(dfs_year_group[year])))
        st_time = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            day_type_cnts = pool.starmap(count_monthly_day_types, zip(dfs_year_group[year], year_list, month_list))
        monthly_day_types_cnt_year_group[year] = {month: day_type_cnt for month, day_type_cnt in zip(month_list, day_type_cnts)}
        print("{} takes {:.1f} sec".format(year, time.time() - st_time))
    return monthly_day_types_cnt_year_group

def add_hour_column(df, year, month):
    st = time.time()
    df['반납_hour'] = df['반납일시'].apply(lambda x: x.hour)
    df['대여_hour'] = df['대여일시'].apply(lambda x: x.hour)
    print("Adding hour column {}-{} takes {:.2f} sec".format(year, month+1, time.time() - st))

def add_day_type_column(df, year, month):
    st = time.time()
    kr_holidays = holidays.KR()
    is_weekend_mask = df['반납일시'].apply(lambda x: x.weekday() > 4)
    is_holiday_mask = df['반납일시'].apply(lambda x: x in kr_holidays)
    is_weekday_mask = ~(is_holiday_mask | is_weekend_mask)
    df.loc[is_weekend_mask, 'day_type'] = '주말'
    df.loc[is_holiday_mask, 'day_type'] = '공휴일'
    df.loc[is_weekday_mask, 'day_type'] = '평일'
    print("Adding day type column {}-{} takes {:.2f} sec".format(year, month+1, time.time() - st))

def group_by_hourly_usage(tgt_df, hour_type):
    assert hour_type in ['반납_hour', '대여_hour']
    hourly_usage_all = tgt_df.groupby([hour_type]).size()
    hourly_usage_day_type = tgt_df.groupby([hour_type, 'day_type']).size()
    tgt_columns = ['hour', 'day_type', 'monthly lent volume']
    multi_indices = [*hourly_usage_day_type.index.tolist(), *[(h, '일일') for h in hourly_usage_all.index.tolist()]]
    hourly_lent_volume_monthly_sum = [*hourly_usage_day_type.values, *hourly_usage_all.values.tolist()]
    tgt_values = [(i[0], i[1], j) for i, j in zip(multi_indices, hourly_lent_volume_monthly_sum)]
    df = pd.DataFrame(tgt_values, columns=tgt_columns)
    df = df.sort_values('hour').reset_index(drop=True)
    return df

def group_by_hourly_usage_all(dfs_year_group, monthly_day_types_cnt_year_group, hour_type):
    assert hour_type in ['반납_hour', '대여_hour']
    hourly_grouped_monthly_df = []
    for year in dfs_year_group.keys():
        st_time = time.time()
        for month in range(len(dfs_year_group[year])):
            df = group_by_hourly_usage(dfs_year_group[year][month], hour_type)
            df['year'] = year
            df['month'] = month + 1
            day_cnt = monthly_day_types_cnt_year_group[year][month]
            for day_type in day_cnt.keys():
                df.loc[df['day_type'] == day_type, 'day_cnt'] = day_cnt[day_type]
            hourly_grouped_monthly_df.append(df)
        print("Group by hour {}year takes {:.1f} sec".format(year, time.time() - st_time))
    hourly_grouped_monthly_df = pd.concat(hourly_grouped_monthly_df)
    return hourly_grouped_monthly_df

def plot_hourly_pattern_bicycle_usage(df, day_cnt, year, month, hour_type):
    assert hour_type in ['반납_hour', '대여_hour']
    xlabel = '반납 시간' if hour_type == '반납_hour' else '대여 시간'
    for tgt_day_type in df['day_type'].unique():
        xs = df[df['day_type'] == tgt_day_type]['hour']
        ys = df[df['day_type'] == tgt_day_type]['monthly lent volume']
        ys /= day_cnt[tgt_day_type]
        plt.plot(xs, ys, label=tgt_day_type)
    plt.title("시간별 자전거 대여량 그래프 ({}-{:02d})".format(year, month + 1))
    plt.xticks(xs, xs)
    plt.xlabel(xlabel)
    plt.ylabel("시간별 일평균 대여량")
    plt.legend()
    plt.show()

def plot_monthly_pattern_bicycle_usage(monthly_mean_df, save_flag=False):
    xs = list(range(1, 13))
    colors = {'일일': '#595959', '평일': 'orange', '주말': 'green', '공휴일': 'red'}
    for tgt_day_type in monthly_mean_df['day_type'].unique():
        tgt_df = monthly_mean_df[monthly_mean_df['day_type'] == tgt_day_type]
        ys = np.array(tgt_df['monthly_mean'].groupby(tgt_df['month']).mean())
        plt.plot(xs, ys, label=tgt_day_type, color=colors[tgt_day_type])
    plt.title("자전거 대여량의 계절성 패턴 (2019년 기준)")
    plt.xticks(xs, xs)
    plt.xlabel("달")
    plt.ylabel("일평균 대여량 (만건)")
    plt.legend()
    if save_flag:
        plt.savefig('4.자전거 대여량의 계절성 패턴.png')
    plt.show()

def plot_yearly_pattern_bicycle_usage(monthly_mean_df):
    xs = monthly_mean_df['year'].unique()
    for tgt_day_type in monthly_mean_df['day_type'].unique():
        tgt_df = monthly_mean_df[monthly_mean_df['day_type'] == tgt_day_type]
        tgt_yearly_sum = np.array(tgt_df['monthly_sum'].groupby(tgt_df['year']).sum())
        tgt_day_cnt = np.array(tgt_df['day_cnt'].groupby(tgt_df['year']).sum())
        ys = tgt_yearly_sum / tgt_day_cnt
        plt.plot(xs, ys, label=tgt_day_type)
    plt.title("일평균 자전거 대여량의 년간 그래프")
    plt.xticks(xs, xs)
    plt.xlabel("년도")
    plt.ylabel("일평균 대여량")
    plt.legend()
    plt.show()

def plot_all_monthly_data(monthly_mean_df):
    xs = np.array(monthly_mean_df.apply(lambda r: str(r['year']) + '-' + str(r['month']), axis='columns'))
    ys = monthly_mean_df['monthly_mean']
    for tgt_day_type in monthly_mean_df['day_type'].unique():
        tgt_row_mask = monthly_mean_df['day_type'] == tgt_day_type
        plt.plot(xs[tgt_row_mask], ys[tgt_row_mask], label=tgt_day_type)
    plt.title("2019~2021 년 일평균 자전거 대여량")
    plt.xticks(rotation=45)
    plt.xlabel("년도-달")
    plt.ylabel("일평균 대여량")
    plt.legend()
    plt.show()

def plot_all_monthly_data_by_year(monthly_mean_df):
    tgt_years = [2019, 2020, 2021]
    for tgt_year in tgt_years:
        tgt_row_mask = (monthly_mean_df['year'] == tgt_year) & (monthly_mean_df['day_type'] == '일일')
        tgt_data = monthly_mean_df.loc[tgt_row_mask, ['month', 'monthly_mean']]
        xs = tgt_data['month']
        ys = tgt_data['monthly_mean']
        plt.plot(xs, ys, label=tgt_year, marker='o')
    plt.title("2019~2021 년 일평균 자전거 대여량")
    plt.xlabel("월")
    plt.ylabel("일평균 대여량")
    plt.legend()
    plt.show()

def plot_hourly_pattern_bicycle_usage_all_year(hourly_grouped_monthly_df_sum, hour_type, save_flag=False):
    assert hour_type in ['반납_hour', '대여_hour']
    type_label = '반납' if hour_type == '반납_hour' else '대여'
    xlabel = '반납 시각' if hour_type == '반납_hour' else '대여 시각'
    colors = {'일일': '#595959', '평일': 'orange', '주말': 'green', '공휴일': 'red'}
    for day_type in hourly_grouped_monthly_df_sum['day_type'].unique():
        tgt_row_mask = hourly_grouped_monthly_df_sum['day_type'] == day_type
        xs = hourly_grouped_monthly_df_sum.loc[tgt_row_mask, 'hour']
        ys = hourly_grouped_monthly_df_sum.loc[tgt_row_mask, 'daily lent volume']
        plt.plot(xs, ys, label=day_type, color=colors[day_type])
    plt.title("자전거 {}량의 시간성 패턴 (2019년 기준)".format(type_label))
    plt.xticks(xs, xs)
    plt.xlabel(xlabel)
    plt.ylabel('일평균 사용량 (건)')
    plt.legend()
    if save_flag:
        plt.savefig("4.자전거 {}량의 시간성 패턴 (2019년 기준).png".format(type_label))
    plt.show()

def plot_lent_vs_return_hourly_bicycle_usage(lent_df, return_df, day_type, save_flag=False):
    lent_df = lent_df[lent_df['day_type'] == day_type]
    return_df = return_df[return_df['day_type'] == day_type]
    lent_hr, lent_volume = lent_df['hour'], lent_df['daily lent volume']
    return_hr, return_volume = return_df['hour'], return_df['daily lent volume']
    plt.plot(lent_hr, lent_volume, label='대여')
    plt.plot(return_hr, return_volume, label='반납')
    plt.title('대여 vs 반납 시간별 자전거 대여량 ({})'.format(day_type))
    plt.xlabel('대여/반납 시각')
    plt.ylabel('일평균 사용량 (건)')
    plt.xticks(lent_hr,lent_hr)
    plt.legend()
    if save_flag:
        plt.savefig('4.대여 vs 반납 시간별 자전거 대여량 ({}).png'.format(day_type))
    plt.show()

def plot_hourly_distance_group(tgt_df, monthly_day_types_cnt, day_type, year, month, save_flag=False):
    if day_type != '일일':
        tgt_df = tgt_df[tgt_df['day_type'] == day_type]
    if monthly_day_types_cnt[day_type] < 1:
        return
    hourly_pairwise_dist_group = tgt_df.groupby(['대여소 거리 그룹', '반납_hour']).size()
    hourly_pairwise_dist_group = hourly_pairwise_dist_group / monthly_day_types_cnt[day_type]
    multi_indices = np.array(hourly_pairwise_dist_group.index.tolist())
    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    graph_labels = ['G1: 0~500m', 'G2: 500~1000m', 'G3: 1000~2000m', 'G4: 2000~4000m', 'G5: 4000m~']
    for i, label in enumerate(labels):
        group_mask = (np.array(multi_indices) == label).any(axis=1)
        xs = multi_indices[group_mask][:, 1].astype(int)
        ys = hourly_pairwise_dist_group.values[group_mask]
        plt.plot(xs, ys, label=graph_labels[i])
    plt.title("시간별 대여-반납 거리 분석 ({}-{:02d} {})".format(year, month, day_type))
    plt.xticks(xs, xs)
    plt.xlabel("반납 시각")
    plt.ylabel("일평균 대여량 (건)")
    plt.legend()
    if save_flag:
        plt.savefig("4.시간별 대여-반납 거리 분석 ({}-{:02d} {}).png".format(year, month, day_type))
    plt.show()

def add_pairwise_dist_columns(df_with_loc):
    start_latitude, start_longitude = df_with_loc['대여 대여소 위도'], df_with_loc['대여 대여소 경도']
    end_latitude, end_longitude = df_with_loc['반납 대여소 위도'], df_with_loc['반납 대여소 경도']
    station_distance_meter = haversine_np(start_longitude, start_latitude, end_longitude, end_latitude)
    df_with_loc['대여소 거리'] = station_distance_meter

def process_pairwise_distance(df, station_loc_df):
    st_time = time.time()
    ## compute pairwise distance in meter
    df_with_loc = merge_lent_amount_with_location_table(station_loc_df, df)
    add_pairwise_dist_columns(df_with_loc)
    ## 대여소 거리별 그룹 나누기
    dist_group1 = df_with_loc['대여소 거리'] < 500
    dist_group2 = (df_with_loc['대여소 거리'] >= 500) & (df_with_loc['대여소 거리'] < 1000)
    dist_group3 = (df_with_loc['대여소 거리'] >= 1000) & (df_with_loc['대여소 거리'] < 2000)
    dist_group4 = (df_with_loc['대여소 거리'] >= 2000) & (df_with_loc['대여소 거리'] < 4000)
    dist_group5 = ~(dist_group1 | dist_group2 | dist_group3 | dist_group4)
    df_with_loc.loc[dist_group1, '대여소 거리 그룹'] = "G1"
    df_with_loc.loc[dist_group2, '대여소 거리 그룹'] = "G2"
    df_with_loc.loc[dist_group3, '대여소 거리 그룹'] = "G3"
    df_with_loc.loc[dist_group4, '대여소 거리 그룹'] = "G4"
    df_with_loc.loc[dist_group5, '대여소 거리 그룹'] = "G5"
    df_with_loc.drop(columns=['대여 대여소 위도', '대여 대여소 경도', '반납 대여소 위도', '반납 대여소 경도'])
    print("Process pairwise distance takes {:.1f} sec".format(time.time() - st_time))
    return df_with_loc

def add_return_hour_and_day_type(df, year, month):
    st_time = time.time()
    add_hour_column(df, year, month)
    add_day_type_column(df, year, month)
    print("{}-{:02d} takes {:.1f} sec".format(year, month+1, time.time() - st_time))
    return df

def add_return_hour_and_day_type_all(dfs_year_group):
    for year in dfs_year_group.keys():
        num_months = len(dfs_year_group[year])
        year_list = [year] * num_months
        month_list = list(range(len(dfs_year_group[year])))
        st_time = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            dfs_year_group[year] = pool.starmap(add_return_hour_and_day_type, zip(dfs_year_group[year], year_list, month_list))
        print("{} takes {:.1f} sec".format(year, time.time()-st_time))

def topk_frequent_station(tgt_df, station_loc_df, topk=10):
    tgt_df_frequent_lent_df = tgt_df[['대여 대여소번호']].groupby('대여 대여소번호').size().reset_index()\
                                .rename(columns={'대여 대여소번호': '대여소 번호', 0: 'count'}).sort_values('count', ascending=False)
    return pd.merge(station_loc_df, tgt_df_frequent_lent_df[0:topk], how='inner').sort_values('count', ascending=False).reset_index(drop=True)

def plot_monthly_sum_all(dfs_year_group):
    each_month_volume_dict = {}
    year_list = list(dfs_year_group.keys())
    for tgt_year in year_list:
        tgt_year_df = dfs_year_group[tgt_year]
        each_month_volume_list = [len(tgt_year_df[i]) / 1e6 for i in range(len(tgt_year_df))]
        each_month_volume_dict[tgt_year] = each_month_volume_list
        month_list = list(range(1, 1 + len(each_month_volume_list)))
        plt.plot(month_list, each_month_volume_list, label=tgt_year + '년', marker='o')
    plt.xticks(list(range(1, 13)), list(range(1, 13)))
    plt.xlabel('월')
    plt.ylabel('월별 총 대여량 (백만)')
    plt.title("월별 총 대여량의 년간 패턴 그래프")
    plt.legend()
    plt.show()

def haversine_arr2arr_v1(src_longs, src_lats, tgt_longs, tgt_lats, thresh):
    bicycle2subway_near_flag_list = []
    for i in range(len(tgt_longs)):
        station_distance_meter = haversine_np(src_longs, src_lats, tgt_longs[i], tgt_lats[i])
        bicycle2subway_near_flag = (station_distance_meter < thresh).any()
        bicycle2subway_near_flag_list.append(bicycle2subway_near_flag)
    return np.stack(bicycle2subway_near_flag_list)

def haversine_arr2arr_v2(src_longs, src_lats, tgt_longs, tgt_lats, thresh):
    src_lats = np.repeat(src_lats.reshape(-1, 1), len(tgt_lats), axis=1)
    src_longs = np.repeat(src_longs.reshape(-1, 1), len(tgt_lats), axis=1)
    bicycle2subway_dist_matrix = haversine_np(src_longs, src_lats, tgt_longs, tgt_lats)
    return (bicycle2subway_dist_matrix < thresh).any(axis=1)

def bicycle2subway_dist(df_loc, subway_loc, thresh=500):
    start_lat, start_long = df_loc['대여 대여소 위도'].values, df_loc['대여 대여소 경도'].values
    end_lat, end_long = df_loc['반납 대여소 위도'].values, df_loc['반납 대여소 경도'].values
    subway_lat_list, subway_long_list = subway_loc['역위도'].values, subway_loc['역경도'].values
    start_lat, start_long, end_lat, end_long = map(lambda x: x.astype(np.float32), [start_lat, start_long, end_lat, end_long])
    subway_lat_list, subway_long_list = map(lambda x: x.astype(np.float32), [subway_lat_list, subway_long_list])
    ## preparation for multiprocessing
    n_cpu = mp.cpu_count()
    thresh_list = [thresh] * n_cpu
    subway_long_sublist = [subway_long_list] * n_cpu
    subway_lat_sublist = [subway_lat_list] * n_cpu
    MAX_ROW_PER_PROCESS = 2e5
    num_loop = math.ceil(len(start_lat) / n_cpu / MAX_ROW_PER_PROCESS)
    start_lat_list = np.array_split(start_lat, num_loop)
    start_long_list = np.array_split(start_long, num_loop)
    end_lat_list = np.array_split(end_lat, num_loop)
    end_long_list = np.array_split(end_long, num_loop)
    bicycle2subway_dist_start_total, bicycle2subway_dist_end_total = [], []
    for loop_idx in range(num_loop):
        ## start bicycle - subway distance
        start_lat_sublist = np.array_split(start_lat_list[loop_idx], n_cpu)
        start_long_sublist = np.array_split(start_long_list[loop_idx], n_cpu)
        start_time = time.time()
        with mp.Pool(processes=n_cpu) as pool:
            bicycle2subway_dist_start_list = pool.starmap(haversine_arr2arr_v2, \
                zip(start_long_sublist, start_lat_sublist,  subway_long_sublist, subway_lat_sublist, thresh_list))
        bicycle2subway_dist_start_list = np.concatenate(bicycle2subway_dist_start_list)
        bicycle2subway_dist_start_total.append(bicycle2subway_dist_start_list)
        print("[{}/{}] 대여소-지하철 처리 {:.1f} 초".format(loop_idx+1, num_loop, time.time() - start_time))
        ## end bicycle - subway distance
        end_lat_sublist = np.array_split(end_lat_list[loop_idx], n_cpu)
        end_long_sublist = np.array_split(end_long_list[loop_idx], n_cpu)
        start_time = time.time()
        with mp.Pool(processes=n_cpu) as pool:
            bicycle2subway_dist_end_list = pool.starmap(haversine_arr2arr_v2, \
                zip(end_long_sublist, end_lat_sublist,  subway_long_sublist, subway_lat_sublist, thresh_list))
        bicycle2subway_dist_end_list = np.concatenate(bicycle2subway_dist_end_list)
        bicycle2subway_dist_end_total.append(bicycle2subway_dist_end_list)
        print("[{}/{}] 반납소-지하철 처리 {:.1f} 초".format(loop_idx+1, num_loop, time.time() - start_time))
    bicycle2subway_dist_start_total = np.concatenate(bicycle2subway_dist_start_total)
    bicycle2subway_dist_end_total = np.concatenate(bicycle2subway_dist_end_total)
    return bicycle2subway_dist_start_total, bicycle2subway_dist_end_total


def label_seoul_zones(bike_stations_map):
    Marker([37.57, 126.82],  icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">강서</div>',)).add_to(bike_stations_map)
    Marker([37.525, 126.85], icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">양천</div>',)).add_to(bike_stations_map)
    Marker([37.525, 126.895], icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">영등포</div>',)).add_to(bike_stations_map)
    Marker([37.5, 126.83], icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">구로</div>',)).add_to(bike_stations_map)
    Marker([37.47, 126.89], icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">금천</div>',)).add_to(bike_stations_map)
    Marker([37.47, 126.93], icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">관악</div>',)).add_to(bike_stations_map)
    Marker([37.51, 126.94], icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0), 
            html='<div style="font-size: 16pt; font-weight:bold">동작</div>',)).add_to(bike_stations_map)
    Marker(
        [37.567, 126.895],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">마포</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.645, 127],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">강북</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.61, 127],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">성북</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.63, 126.915],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">은평</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.61, 127.085],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">중랑</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.58, 126.912],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">서대문</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.59, 126.97],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">종로</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.535, 126.97],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">용산</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.565, 126.99],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">중</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.5, 126.99],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">서초</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.52, 127.1],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">송파</div>',
            )
        ).add_to(bike_stations_map)

    Marker(
        [37.56, 127.14],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">강동</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.56, 127.08],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">광진</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.56, 127.03],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">성동</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.586, 127.035],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">동대문</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.642, 127.065],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">노원</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.66, 127.025],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">도봉</div>',
            )
        ).add_to(bike_stations_map)
    Marker(
        [37.52, 127.03],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html='<div style="font-size: 16pt; font-weight:bold">강남</div>',
            )
        ).add_to(bike_stations_map)