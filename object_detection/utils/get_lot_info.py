from cassandra.cluster import Cluster
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("lotno")
    return parser.parse_args()


def getlotinfo(session, lotno):
    return session.execute(
        """
        SELECT
            lotno,
            basedir,
            chip_width,
            chip_height,
            chipdir,
            create_dt,
            machine,
            mtype,
            register_dt,
            ringid,
            summary
        FROM lots WHERE lotno=%(lotno)s
        """,
        {
            "lotno": lotno
        }
    ).one()


def get_chipinfo(session, mtype):
    return session.execute(
        """
        SELECT
            chip_width,
            chip_height
        FROM lots WHERE mtype=%(mtype)s
        ALLOW FILTERING
        """,
        {
            "mtype": mtype
        }
    ).one()


def get_chipsize(mtype, nodes=["ai-ed800-2.nichia.local"],
                 keyspace="b238_2109"):
    cluster = Cluster(nodes)
    session = cluster.connect(keyspace)
    row = get_chipinfo(session, mtype)
    return row.chip_width, row.chip_height


def main(lotno,
         nodes=["ai-ed800-2.nichia.local"],
         keyspace="b238_2109"):
    cluster = Cluster(nodes)
    session = cluster.connect(keyspace)

    row = getlotinfo(session, lotno)
    return row


if __name__ == "__main__":
    main(**vars(get_args()))
