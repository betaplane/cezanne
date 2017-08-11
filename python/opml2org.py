#!/usr/bin/env python
import xml.etree.ElementTree as et

t = et.parse('../../notes/namelist.opml')
b = t.find('body')
with open('../../notes/namelist.org', 'w') as f:
    for e in b.iter():
        if len(e.attrib) == 0:
            continue
        if 'meaning' in e.attrib:
            f.write('|{}|{}|\n'.format(e.attrib['text'], e.attrib['meaning']))
        else:
            f.write('* {}\n'.format(e.attrib['text']))
