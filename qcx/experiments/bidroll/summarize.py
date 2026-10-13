import json, sys
for f in sys.argv[1:]:
  g = json.load(open(f))['competitive_promotion_gate']
  pw = g.get('pairwise', {}); pa = g.get('placement_advantage', {}); sa = g.get('official_score_advantage', {})
  cp = g.get('candidate_own_paradox', {}); rp = g.get('reference_personal_paradox', {}); pr = g.get('candidate_prediction', {})
  print(f.split('/')[-1], g['candidate'], 'deals', g['independent_deal_blocks'], 'matches', g['matches'])
  print('  mean placement', g['mean_official_placement'], 'first-place share', g['first_place_share'])
  print('  pairwise', {k: v for k, v in pw.items() if not isinstance(v, (list, dict))})
  print('  placement_adv', {k: v for k, v in pa.items() if not isinstance(v, (list, dict))})
  print('  score_adv', {k: v for k, v in sa.items() if not isinstance(v, (list, dict))})
  print('  own paradox', {k: v for k, v in cp.items() if not isinstance(v, (list, dict))}, 'ref', {k: v for k, v in rp.items() if not isinstance(v, (list, dict))})
  print('  prediction', {k: v for k, v in pr.items() if not isinstance(v, (list, dict))})
