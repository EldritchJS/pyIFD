## pyIFD - Python-based Image Forgery Detection Toolkit
As photo-editing tools become more ubiquitous and easier to use, it becomes equally important to detect these manipulations.
This is especially key when the edits are malicious.
While toolkits for this purpose exist [1], they are often not well maintained nor easy to integrate and use, besides being written in proprietary programming language.
The goal of this package is to reduce the barrier to entry for people to detect image manipulations on their own.

The initial version of this toolkit is based on [1].
New techniques of image forgery detection will soon be added to it.

[1]: [Image Forensics Toolkit](https://github.com/MKLab-ITI/image-forensics)

### Algorithms
The toolkit consists of a number of different algorithms (click on each to be taken to a usage page and description).
- [ADQ1](pdocs/pyIFD/ADQ1.md)
- [ADQ2](pdocs/pyIFD/ADQ2.md)
- [ADQ3](pdocs/pyIFD/ADQ3.md)
- [BLK](pdocs/pyIFD/BLK.md)
- [CAGI](pdocs/pyIFD/CAGI.md)
- [CFA1](pdocs/pyIFD/CFA1.md)
- [DCT](pdocs/pyIFD/DCT.md)
- [ELA](pdocs/pyIFD/ELA.md)
- [GHOST](pdocs/pyIFD/GHOST.md)
- [NADQ](pdocs/pyIFD/NADQ.md)
- [NOI1](pdocs/pyIFD/NOI1.md)
- [NOI2](pdocs/pyIFD/NOI2.md)
- [NOI4](pdocs/pyIFD/NOI4.md)
- [NOI5](pdocs/pyIFD/NOI5.md)

### References
- [ALL](https://link.springer.com/content/pdf/10.1007/s11042-016-3795-2.pdf)
- [ADQ1](https://www.sciencedirect.com/science/article/pii/S0031320309001198?casa_token=RYKfNwIS8WYAAAAA:BoG6UVqSJIbCO28z4K4UBrMplCP0fFt76wM69E8n6dy7e65t7X16xbwhbbfcbXQwrya0ujQitvg)
- [ADQ2](https://ieeexplore.ieee.org/iel5/5916934/5946226/05946978.pdf?casa_token=HbhOFnI7LxkAAAAA:ukrMyrxZ4Pkbvgnx2pO4JTHbHE6oHCf7ku-v9RhLD3LfYyPWarh1fIdONrK4WE3WudV1nN932A)
- [ADQ3](https://ieeexplore.ieee.org/iel7/7070475/7084286/07084318.pdf?casa_token=0csGlsul6S8AAAAA:kNv63mbMnOcMqv27tiockLMeNpQTzDiypx7hEwIB_BH-RXdWkvHh3Cf9OjEOgc5lO78fThalVQ)
- [BLK](https://www.sciencedirect.com/science/article/pii/S0165168409001315?casa_token=zVRxPnKzIAYAAAAA:WdiJ8fJay9WRZv_5ckljkMaQCJCUCaCMS6x84qNsHJTDrTSrJWIK1IJXDGwKZgkr9g6E1Y-s1X0)
- [CAGI](https://www.sciencedirect.com/science/article/pii/S1047320318301068?casa_token=EJIQ0I589HUAAAAA:r_n-GvI9MMbcYG9Et8rnLu4uA3bffHs1zJgEpVoV6Rem1yfzbEOa2zQ1PtwWythcUyroMNyDEo4)
- [CFA1](https://ieeexplore.ieee.org/iel5/10206/4358835/06210378.pdf?casa_token=KMcMGB4zSRYAAAAA:hjBeyV2wUQOT7WTsN_ysH1yWZOGpiIThEBOVYOT-FL8gyByDJ0Zgn1QRUQcq-LcozyFhzaj5vw)
- [DCT](https://ieeexplore.ieee.org/iel5/4284552/4284553/04284574.pdf?casa_token=WXhr8eg6d2UAAAAA:VMKd4QfSj9qBYSdclf_QrmHDaxN3GSA0w7Vp5wK_CLadD5w0KEcsT5OpWeH7mS1Mc0VL3xflmQ)
- [ELA](https://www.blackhat.com/presentations/bh-usa-07/Krawetz/Whitepaper/bh-usa-07-krawetz-WP.pdf)
- [GHOST](https://ieeexplore.ieee.org/iel5/10206/4782049/04773149.pdf?casa_token=dFM-stUDQiQAAAAA:SEetzIaeQuKQXfmIMEkpSW1QgK8vU8uhUCLbh3obcHouMmJSrxwNBv_guOpHmkM04SKVGxeEog)
- [NADQ](https://ieeexplore.ieee.org/iel5/10206/4358835/06151134.pdf?casa_token=8HV81UW33moAAAAA:-cwoVCG1MkZ_pZA4SwclVYrg3WQ1BZInhbzCvlhaQkIJop8xxBMKxadMrAaQDV1xrRFVTF62Vg)
- [NOI1](https://www.sciencedirect.com/science/article/pii/S0262885609000146?casa_token=vofFPS05_mgAAAAA:_AsHy_iuyYr22u1pVck9T0PLFg0t54rOndNXkUSJtBttpKSavTYDLsVNVbMD88Ld4mWcNyyuVpQ)
- [NOI2](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s11263-013-0688-y.pdf&casa_token=SWn_1aK9uYwAAAAA:Q8LVOsV-ISkhaF09WnrbfaDliOq9U5V23Zc0NG9UuVGSwWa1S7uJzLXHUZoVVPT_9OLOHDQ0LE8Vci43gw)
- [NOI4](https://29a.ch/2015/08/21/noise-analysis-for-image-forensics)
- [NOI5](http://www.escience.cn/people/Zenghui/index.html)


### Authors
The University of Notre Dame Computer Vision Research Laboratory ([CVRL](https://cvrl.nd.edu/)) and Red Hat Artificial Intelligence Center of Excellence.

### Installation
`pip install git+https://github.com/eldritchjs/pyIFD`

### Generating documentation
`pdoc3 --force -o ./pdocs ./src/pyIFD`

### Running tests
`pytest`
