#!/usr/bin/env python

"""
ロットNoからマップ情報
"""

import pyodbc
import os
import re


def get_sql_conn(instance, user, password, db):
    """sqlserverセッションを取得
    Parameters
    ----------

    Returns
    -------
    """

    connstr = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=" + instance + ";uid=" + user + ";pwd=" + password + ";DATABASE=" + db
    return pyodbc.connect(connstr)


def get_shape(lotno, instance="hqdb1\\inst1", user="b238reader", password="AX7uC(aX", db="rslddb"):
    """ロットNoから配置形状

    Parameters
    ----------
    conn :
        sqlserver コネクション
    lotno : str
        ロットNo

    Returns
    -------
    dice_size : (float, float)
        ダイスサイズ(um)
    wafer_center_address : (float, float)
        ウェハ中心アドレス
    wafer_size_cd : str
        ウェハサイズCD (3:4inches, 2:3inches, 1:2inches)
    map_id : int
        マップID
    """

    sql = '''
        select
            sort_lotno
            , shape 配置形状CD
            , memo [配置形状(名)]
        from RSLDDB.dbo.DTTGSORTLOTINFO a
        inner join Process.dbo.TM_Wafer_Quality_Col as b on b.quality_no = 48 and b.quality_cd = a.shape
        where sort_lotno = ?
    '''
    conn = get_sql_conn(instance, user, password, db)
    cursor = conn.cursor()
    cursor.execute(sql, lotno)
    rows = cursor.fetchall()
    cursor.close()
    return rows


def get_dice_size(mtype, instance="hqdb1\\inst1", user="b238reader", password="AX7uC(aX", db="rslddb"):

    sql = '''
        select distinct g.dice_code, z.ダイス_Xサイズ H, z.ダイス_Yサイズ W from process.dbo.TM_マップID m
        inner join process.dbo.TM_フォトリソ組み合わせ p on p.マスク組み合わせCD = m.マスク組み合わせCD
        inner join process.dbo.TM_マーカー座標 z on z.マスクCD_フォトリソG = p.マスクCD_フォトリソG
        inner join 生産在庫管理SQL..GTMPrm g on g.map_id = m.マップID
        where g.dice_code = ?
    '''
    conn = get_sql_conn(instance, user, password, db)
    cursor = conn.cursor()
    cursor.execute(sql, mtype)
    rows = cursor.fetchall()
    cursor.close()
    mtype, chip_height, chip_width = rows[0]
    return chip_width, chip_height


def get_waferspec(lotno, instance="hqdb1\\inst1", user="b238reader", password="AX7uC(aX", db="rslddb"):
    """ロットNoからマップ情報

    Parameters
    ----------
    conn :
        sqlserver コネクション
    lotno : str
        ロットNo

    Returns
    -------
    dice_size : (float, float)
        ダイスサイズ(um)
    wafer_center_address : (float, float)
        ウェハ中心アドレス
    wafer_size_cd : str
        ウェハサイズCD (3:4inches, 2:3inches, 1:2inches)
    map_id : int
        マップID
    """

    sql = '''
        select
            distinct
            z.ダイス_Xサイズ,
            z.ダイス_Yサイズ,
            z.X_基点座標,
            z.Y_基点座標,
            z.X_中心座標,
            z.Y_中心座標,
            z.オリフラ逆までの距離,
            z.オリフラまでの距離,
            s.使用可能半径,
            s.使用可能なオリフラまでの距離,
            z.オリフラ長,
            p.ウェハーサイズCD,
            m.マップID
        from Process.dbo.T_Wafer_Quality q
        inner join Process.dbo.TM_マップID m on m.マップID = q.quality_val8
        inner join Process.dbo.TM_使用可能領域 s on s.使用可能領域CD = m.使用可能領域CD
        inner join Process.dbo.TM_フォトリソ組み合わせ p on p.マスク組み合わせCD = m.マスク組み合わせCD
        inner join Process.dbo.TM_マーカー座標 z on z.マスクCD_フォトリソG = p.マスクCD_フォトリソG
        inner join RSLDDB.dbo.DTTSORTLOT d on d.lot_no = q.lot_no and d.del_flg = 0
        where d.sortlot_no = ?
        --where q.lot_no = ?
        '''
    dummy_data = {
        "R09FRZ5": (1400, 1400),
        "R0770DI": (1400, 1400),
        "R0113YU": (1400, 1400),
        "R0A6MC5": (1000, 1000),
        "R04TGMQ": (1000, 1000),
    }
    if lotno in dummy_data:
        return dummy_data[lotno]
    conn = get_sql_conn(instance, user, password, db)
    cursor = conn.cursor()
    cursor.execute(sql, lotno)
    rows = cursor.fetchall()
    cursor.close()
    if len(rows) != 1:
        raise Exception
    dicesize = rows[0][1], rows[0][0]
    """wafer_center_address = rows[0][4], rows[0][5]
    wafer_size_cd = rows[0][11]
    map_id = rows[0][12]"""
    return dicesize


def get_dice_size2(mtype, instance="hqdb1\\inst1", user="b238reader", password="AX7uC(aX", db="rslddb"):
    sql = '''
    select
        gthick_cd, gthick_ja
    from
        ROOTSDB.dbo.RTM_MVMTYPE
    where
        type_cd = ?
    '''
    conn = get_sql_conn(instance, user, password, db)
    cursor = conn.cursor()
    cursor.execute(sql, mtype)
    rows = cursor.fetchall()
    cursor.close()
    gthick = rows[0][1]
    gthick = re.search(r'\d+', gthick)
    print(gthick.group())


def main():
    get_dice_size2('D0RM-AMA01-22')


if __name__ == "__main__":
    main()
