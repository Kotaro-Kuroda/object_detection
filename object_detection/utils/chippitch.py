#!/usr/bin/env python

"""
型番から粘着シート品名、最小/最大チップ数、ダイス間隔を抽出する
"""

import pyodbc


def get_sql_conn(instance, user, password, db):
    """sqlserverセッションを取得
    Parameters
    ----------

    Returns
    -------
    """

    connstr = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=" + instance + ";uid=" + user + ";pwd=" + password + ";DATABASE=" + db
    return pyodbc.connect(connstr)


def get_chippitch(mtype, instance="hqdb1\\inst1", user="b238reader", password="AX7uC(aX", db="process"):
    """型番から粘着シート品名、最小/最大チップ数、ダイス間隔を抽出する
    Parameters
    ----------
    mtype: 型番

    Returns
    -------
     0: 型番
     1: 粘着シート品名
     2: 装置名
     3: 最小搭載チップ数(社外品)
     4: トレイフル数(四角形)
     5: トレイフル数(φ75mm円形)
     6: トレイフル数(φ90mm円形)
     7: トレイフル数(φ100mm円形)
     8: ダイス間隔(X座標)
     9: ダイス間隔(Y座標)
    10: トレイフル数(四角形・ウレタン時)
    11: トレイフル数(φ75mm円形・ウレタン時)
    12: トレイフル数(φ90mm円形・ウレタン時)
    13: トレイフル数(φ100mm円形・ウレタン時)
    14: ダイス間隔(X座標・ウレタン時)
    15: ダイス間隔(Y座標・ウレタン時)
    """

    sql = '''
declare @型番 VarChar(20)
set @型番 = ?

declare @result varchar(20)

BEGIN
set @result = null

select distinct
  @result = count(*)
from
  rootsdb..rtmtype A with(nolock) inner join
  rootsdb..rtmmconv B with(nolock) on A.material_cd = B.material_cd inner join
  Quality..DTMGBUNLAYOUT C with(nolock) on A.tpdicesize_cd = C.size_cd and A.tpdevice_cd =C.dev_cd inner join
  Quality..DTMGBUNLAYOUTDET D on C.arrangesort_cd = D.arrangesort_cd
where
  B.mtralbase_cd = @型番

  IF @result = '0'
        SELECT DISTINCT
        A.m_material_cd AS "型番"
        ,case B.die_bond_sheet_cd
        when 1 then 'V8S'
        when 2 then 'D371W'
        when 3 then 'V8L'
        else '不明' end AS "粘着シート品名"
        , H.machine_nm "装置名"
        , E.so_minchip as "最小搭載チップ数(社外品)"
        , G.trayfullnum_ct as "トレイフル数(四角形)"
        , G.trayfullnum_circle_ct as "トレイフル数(φ75mm円形)"
        , G.trayfullnum_90circle_ct as "トレイフル数(φ90mm円形)"
        , G.trayfullnum_100circle_ct as "トレイフル数(φ100mm円形)"
        , G.pchspcx_ct as "ダイス間隔(X座標)"
        , G.pchspcy_ct as "ダイス間隔(Y座標)"
        , I.trayfullnum_ct as "トレイフル数(四角形・ウレタン時)"
        , I.trayfullnum_circle_ct as "トレイフル数(φ75mm円形・ウレタン時)"
        , I.trayfullnum_90circle_ct as "トレイフル数(φ90mm円形・ウレタン時)"
        , I.trayfullnum_100circle_ct as "トレイフル数(φ100mm円形・ウレタン時)"
        , I.pchspcx_ct as "ダイス間隔(X座標・ウレタン時)"
        , I.pchspcy_ct as "ダイス間隔(Y座標・ウレタン時)"
        FROM 生産在庫管理SQL..DTMMATERIAL A with(nolock)
        INNER JOIN 生産在庫管理SQL..GTMPrm B with(nolock) ON A.a_material_cd = B.dice_code
        INNER JOIN rootsdb..rtmmconv C with(nolock) ON A.a_material_cd = C.mtralbase_cd
        INNER JOIN rootsdb..rtmtype D with(nolock) ON C.material_cd = D.material_cd
        INNER JOIN quality..dtmgchipchk E with(nolock) ON D.tpdicesize_cd = E.size_cd and D.tpwafer_cd =E.w_size_cd
        INNER JOIN Quality..DTMGBUNLAYOUT F with(nolock) ON D.tpdicesize_cd = F.size_cd and F.dev_cd = ''
        INNER JOIN Quality..DTMGBUNLAYOUTDET G with(nolock) ON F.arrangesort_cd = G.arrangesort_cd
        INNER JOIN RSLDDB..DTMMAC H with(nolock) ON F.mckind_cd = H.machine_cd
        LEFT JOIN Quality..DTMGBUNLAYOUTDET I ON F.arrangesort_cd2 = I.arrangesort_cd
        WHERE A.m_material_cd = @型番
        AND B.die_bond_sheet_cd > 0
        AND
        (
        SELECT
         count(DISTINCT B2.die_bond_sheet_cd)
         FROM 生産在庫管理SQL..DTMMATERIAL A2 with(nolock)
        INNER JOIN 生産在庫管理SQL..GTMPrm B2 with(nolock) ON A2.a_material_cd = B2.dice_code
        WHERE A2.m_material_cd = @型番 and B2.die_bond_sheet_cd <> 0 and B2.die_bond_sheet_cd is not null
        ) = '1'

    ELSE
        SELECT DISTINCT
        A.m_material_cd AS "型番"
        ,case B.die_bond_sheet_cd
        when 1 then 'V8S'
        when 2 then 'D371W'
        when 3 then 'V8L'
        else '不明' end AS "粘着シート品名"
        , H.machine_nm "装置名"
        , E.so_minchip as "最小搭載チップ数(社外品)"
        , G.trayfullnum_ct as "トレイフル数(四角形)"
        , G.trayfullnum_circle_ct as "トレイフル数(φ75mm円形)"
        , G.trayfullnum_90circle_ct as "トレイフル数(φ90mm円形)"
        , G.trayfullnum_100circle_ct as "トレイフル数(φ100mm円形)"
        , G.pchspcx_ct as "ダイス間隔(X座標)"
        , G.pchspcy_ct as "ダイス間隔(Y座標)"
        , I.trayfullnum_ct as "トレイフル数(四角形・ウレタン時)"
        , I.trayfullnum_circle_ct as "トレイフル数(φ75mm円形・ウレタン時)"
        , I.trayfullnum_90circle_ct as "トレイフル数(φ90mm円形・ウレタン時)"
        , I.trayfullnum_100circle_ct as "トレイフル数(φ100mm円形・ウレタン時)"
        , I.pchspcx_ct as "ダイス間隔(X座標・ウレタン時)"
        , I.pchspcy_ct as "ダイス間隔(Y座標・ウレタン時)"
        FROM 生産在庫管理SQL..DTMMATERIAL A with(nolock)
        INNER JOIN 生産在庫管理SQL..GTMPrm B with(nolock) ON A.a_material_cd = B.dice_code
        INNER JOIN rootsdb..rtmmconv C with(nolock) ON A.a_material_cd = C.mtralbase_cd
        INNER JOIN rootsdb..rtmtype D with(nolock) ON C.material_cd = D.material_cd
        INNER JOIN quality..dtmgchipchk E with(nolock) ON D.tpdicesize_cd = E.size_cd and D.tpwafer_cd =E.w_size_cd
        INNER JOIN Quality..DTMGBUNLAYOUT F with(nolock) ON D.tpdicesize_cd = F.size_cd and D.tpdevice_cd =F.dev_cd
        INNER JOIN Quality..DTMGBUNLAYOUTDET G with(nolock) ON F.arrangesort_cd = G.arrangesort_cd
        INNER JOIN RSLDDB..DTMMAC H with(nolock) ON F.mckind_cd = H.machine_cd
        LEFT JOIN Quality..DTMGBUNLAYOUTDET I ON F.arrangesort_cd2 = I.arrangesort_cd
        WHERE A.m_material_cd = @型番
        AND B.die_bond_sheet_cd > 0
        AND
        (
        SELECT
         count(DISTINCT B2.die_bond_sheet_cd)
         FROM 生産在庫管理SQL..DTMMATERIAL A2 with(nolock)
        INNER JOIN 生産在庫管理SQL..GTMPrm B2 with(nolock) ON A2.a_material_cd = B2.dice_code
        WHERE A2.m_material_cd = @型番 and B2.die_bond_sheet_cd <> 0 and B2.die_bond_sheet_cd is not null
        ) = '1'
END;
        '''

    conn = get_sql_conn(instance, user, password, db)
    cursor = conn.cursor()
    cursor.execute(sql, mtype)
    rows = cursor.fetchall()
    cursor.close()
    if len(rows) == 0:
        raise Exception
    return rows


if __name__ == "__main__":
    chip_pitch_info = get_chippitch("4HB-HE70NC")
    print(chip_pitch_info)
