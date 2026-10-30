/* Percentile dice pairing: one d00 (tens) plus one d10 (units) form a d100 roll.
   Pure so it can be exercised without a DOM. */
(function (root) {
  function center(die) {
    if (Array.isArray(die.bbox) && die.bbox.length === 4) {
      return [(die.bbox[0] + die.bbox[2]) / 2, (die.bbox[1] + die.bbox[3]) / 2];
    }
    return null;
  }

  function pairPercentile(dice) {
    const tens = [];
    const units = [];
    dice.forEach((die, index) => {
      if (!Number.isInteger(die.value)) return;
      if (die.kind === 'd00' && die.value >= 0 && die.value <= 90) tens.push(index);
      else if (die.kind === 'd10' && die.value >= 1 && die.value <= 10) units.push(index);
    });
    const used = new Set();
    const pairs = [];
    tens.forEach(tensIndex => {
      let best = -1;
      let bestScore = Infinity;
      const tensCenter = center(dice[tensIndex]);
      units.forEach(unitIndex => {
        if (used.has(unitIndex)) return;
        let score = unitIndex;
        const unitCenter = center(dice[unitIndex]);
        if (tensCenter && unitCenter) {
          score = Math.hypot(tensCenter[0] - unitCenter[0], tensCenter[1] - unitCenter[1]);
        }
        if (score < bestScore) {
          bestScore = score;
          best = unitIndex;
        }
      });
      if (best < 0) return;
      used.add(best);
      // A d10 showing 0 is stored as 10; in a percentile pair it means 0 units.
      const raw = dice[tensIndex].value + (dice[best].value % 10);
      pairs.push({ tens: tensIndex, units: best, total: raw === 0 ? 100 : raw });
    });
    return { pairs, usedUnits: used };
  }

  if (typeof module === 'object' && module.exports) module.exports = { pairPercentile };
  else root.pairPercentile = pairPercentile;
})(typeof globalThis !== 'undefined' ? globalThis : this);
