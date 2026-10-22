"""Approximate country/territory centroids for display, not logistics GPS."""

_RAW = """
Afganistán|33|65
Albania|41|20
Alemania|51|9
Angola|-12.5|18.5
Arabia Saudí|25|45
Argelia|28|3
Argentina|-34|-64
Armenia|40|45
Australia|-27|133
Austria|47.33333|13.33333
Azerbaiyán|40.5|47.5
Bangladés|24|90
Barbados|13.16667|-59.53333
Baréin|26|50.55
Belice|17.25|-88.75
Benín|9.5|2.25
Bielorrusia|53|28
Bolivia|-17|-65
Bosnia y Herzegovina|44|18
Botsuana|-22|24
Brasil|-10|-55
Bulgaria|43|25
Burkina Faso|13|-2
Burundi|-3.5|30
Bután|27.5|90.5
Bélgica|50.83333|4
Camboya|13|105
Camerún|6|12
Canada|60|-95
Chad|15|19
Chile|-30|-71
China|35|105
Chipre|35|33
Colombia|4|-72
Corea del Sur|37|127.5
Costa Rica|10|-84
Costa de Marfil|8|-5
Croacia|45.16667|15.5
Cuba|21.5|-80
Dinamarca|56|10
Ecuador|-2|-77.5
Egipto|27|30
El Salvador|13.83333|-88.91667
Emiratos Árabes Unidos|24|54
Eritrea|15|39
Eslovaquia|48.66667|19.5
Eslovenia|46.11667|14.81667
España|40|-4
Estados Unidos|38|-97
EE. UU.|38|-97
Estonia|59|26
Etiopía|8|38
Filipinas|13|122
Finlandia|64|26
Francia|46|2
Gabón|-1|11.75
Georgia|42|43.5
Ghana|8|-2
Grecia|39|22
Guadalupe|16.25|-61.58333
Guatemala|15.5|-90.25
Guayana Francesa|4|-53
Guinea|11|-10
Guinea Ecuatorial|2|10
Guinea-Bissau|12|-15
Guyana|5|-59
Haití|19|-72.41667
Hong Kong|22.25|114.16667
Honduras|15|-86.5
Hungría|47|20
India|20|77
Indonesia|-5|120
Irak|33|44
Irlanda|53|-8
Irán|32|53
Israel|31.5|34.75
Italia|42.83333|12.83333
Jamaica|18.25|-77.5
Japón|36|138
Jordania|31|36
Kazajistán|48|68
Kenia|1|38
Kirguistán|41|75
Kuwait|29.5|45.75
Laos|18|105
Lesoto|-29.5|28.5
Liberia|6.5|-9.5
Libia|25|17
Lituania|56|24
Luxemburgo|49.75|6.16667
Líbano|33.83333|35.83333
Macedonia|41.83333|22
Madagascar|-20|47
Malasia|2.5|112.5
Mali|17|-4
Marruecos|32|-5
Martinica|14.66667|-61
Mauritania|20|-12
Moldavia|47|29
Mongolia|46|105
Montenegro|42.5|19.3
Mozambique|-18.25|35
Myanmar (Birmania)|21.9|95.95
México|23|-102
Namibia|-22|17
Nepal|28|84
Nicaragua|13|-85
Nigeria|10|8
Noruega|62|10
Nueva Zelanda|-41|174
Níger|16|8
Omán|21|57
Pakistán|30|70
Panamá|9|-80
Papúa Nueva Guinea|-6|147
Paraguay|-23|-58
Países Bajos|52.5|5.75
Perú|-10|-76
Polonia|52|20
Portugal|39.5|-8
Qatar|25.5|51.25
Reino Unido|54|-2
República Centroafricana|7|21
República Checa|49.75|15.5
República Democrática del Congo|0|25
República Dominicana|19|-70.66667
República de Gambia|13.46667|-16.56667
República del Congo|-1|15
Ruanda|-2|30
Rumania|46|25
Rusia|60|100
Senegal|14|-14
Serbia|44|21
Sierra Leona|8.5|-11.5
Singapur|1.36667|103.8
Siria|35|38
Somalia|10|49
Sri Lanka|7|81
Suazilandia|-26.5|31.5
SudAfrica|-29|24
Sudán|15|30
Sudán del Sur|7|30
Suecia|62|15
Suiza|47|8
Surinam|4|-56
Sáhara Occidental|24.5|-13
Tailandia|15|100
Taiwán|23.5|121
Tanzania|-6|35
Tayikistán|39|71
Togo|8|1.16667
Trinidad y Tobago|11|-61
Turkmenistán|40|60
Turquía|39|35
Túnez|34|9
Ucrania|49|32
Uganda|1|32
Uruguay|-33|-56
Uzbekistán|41|64
Venezuela|8|-66
Vietnam|16.16667|107.83333
Yemen|15|48
Yibuti|11.5|43
Zambia|-15|30
Zimbabue|-20|30
"""
COUNTRY_COORDS = {}
for _line in _RAW.strip().splitlines():
    _name, _lat, _lon = _line.split("|")
    COUNTRY_COORDS[_name] = (float(_lat), float(_lon))
del _RAW, _line, _name, _lat, _lon
