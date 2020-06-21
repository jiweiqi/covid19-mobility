
# Inspired by https://gist.github.com/JeffPaine/3083347
# Access full state names using us_states.keys()
# Access all state abbreviations using us_states.values()
us_states = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    # 'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

for state in us_states.keys():
    state_abbr = us_states[state].lower()
    print(state.lower())
    filename = 'projection/us-' + state_abbr + '.md'
    print(filename)

    with open(filename, 'w') as the_file:
        the_file.write('---\n')
        the_file.write('layout: default\n')
        the_file.write('permalink: /projection/us-'+state_abbr+'\n')
        the_file.write('---\n')
        the_file.write('\n')
        the_file.write('### '+state+' Fuel Demand Projection\n')
        the_file.write('\n')
        the_file.write('<p align="center">\n')
        the_file.write('    {% include html/us_' +
                       state.replace(" ", "_")+'_fuel_demand.html %}\n')
        # the_file.write('    {% include html/us_'+state.replace(" ", "_")+'_mobility_Grocery_and_Pharmacy.html %}\n')
        # the_file.write('    {% include html/us_'+state.replace(" ", "_")+'_mobility_Parks.html %}\n')
        # the_file.write('    {% include html/us_'+state.replace(" ", "_")+'_mobility_Retail_and_Recreation.html %}\n')
        # the_file.write('    {% include html/us_'+state.replace(" ", "_")+'_mobility_Workplaces.html %}\n')
        the_file.write('</p>\n')
