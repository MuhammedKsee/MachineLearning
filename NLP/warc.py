from warcio import ArchiveIterator

with open('CC-MAIN-20230331120102-20230331150102-00000.warc.gz', 'rb') as fileobj:
    for record in ArchiveIterator(fileobj):
        print(record)
