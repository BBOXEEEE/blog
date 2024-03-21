import React from "react";

import { useSiteMetadata, useCategoriesList } from "@/hooks";

import { Author } from "./Author";
import { Contacts } from "./Contacts";
import { Copyright } from "./Copyright";
import { Menu } from "./Menu";
import { Category } from "./Category";

import * as styles from "./Sidebar.module.scss";

type Props = {
  isIndex?: boolean;
};

const Sidebar = ({ isIndex }: Props) => {
  const { author, copyright, menu } = useSiteMetadata();
  const categoriesList = useCategoriesList();
  
  const category = categoriesList.map(cat => ({
    label: cat.fieldValue,
    path: `/category/${cat.fieldValue.toLowerCase().replace(/\s+/g, '-')}`,
    count : cat.totalCount
  }));

  return (
    <div className={styles.sidebar}>
      <div className={styles.inner}>
        <Author author={author} isIndex={isIndex} />
        <Contacts contacts={author.contacts} />
        <Menu menu={menu} />
        <Category category={category} />
        <Copyright copyright={copyright} />
      </div>
    </div>
  );
};

export default Sidebar;
